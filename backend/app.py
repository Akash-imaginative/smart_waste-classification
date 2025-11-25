from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from collections import Counter
import requests
from math import radians, cos, sin, asin, sqrt

# --- App setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# --- Model loading ---
yolo_model = YOLO('yolov5s.pt')
cnn_model = tf.keras.models.load_model(r"C:\Users\Lenovo\OneDrive\Desktop\Waste-Classification\models\densenet121_waste_classifier.keras")

cnn_class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# --- Utilities ---
def preprocess_crop(crop_img):
    crop_img = cv2.resize(crop_img, (224, 224))
    crop_img = crop_img.astype("float32") / 255.0
    crop_img = img_to_array(crop_img)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img

def annotate_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    all_detections = []
    
    # BATCH OPTIMIZATION: Collect all crops first, then predict in one batch
    yolo_crops_data = []  # Store (crop, bbox, crop_image) for YOLO detections
    
    # STEP 1: YOLO for distinct objects - COLLECT CROPS
    try:
        results = yolo_model(rgb_image, conf=0.12, iou=0.40, max_det=100, verbose=False)
        if results and results[0].boxes:
            yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in yolo_boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                
                if (xmax - xmin) < 30 or (ymax - ymin) < 30:
                    continue
                
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)
                
                crop = image[ymin:ymax, xmin:xmax]
                if crop.size == 0:
                    continue
                
                yolo_crops_data.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'crop': crop
                })
    except Exception as e:
        print(f"YOLO detection failed: {e}")
    
    # BATCH PREDICT for YOLO crops
    if yolo_crops_data:
        batch_crops = np.vstack([preprocess_crop(item['crop']) for item in yolo_crops_data])
        batch_preds = cnn_model.predict(batch_crops, verbose=0)
        
        for idx, (preds, item) in enumerate(zip(batch_preds, yolo_crops_data)):
            crop = item['crop']
            bbox = item['bbox']
            
            top_2_indices = np.argsort(preds)[-2:][::-1]
            pred_index = top_2_indices[0]
            confidence = float(preds[pred_index])
            second_confidence = float(preds[top_2_indices[1]])
            confidence_margin = confidence - second_confidence
            
            # Fix bias: Penalize shoes/clothes predictions that don't look like fabric
            predicted_label = cnn_class_names[pred_index]
            if predicted_label in ['shoes', 'clothes']:
                gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                texture_std = np.std(gray_check)
                
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hsv_std = np.mean([np.std(hsv_crop[:,:,0]), np.std(hsv_crop[:,:,1]), np.std(hsv_crop[:,:,2])])
                
                if texture_std < 35 or hsv_std < 40:
                    top_3_indices = np.argsort(preds)[-3:][::-1]
                    for alt_idx in top_3_indices[1:]:
                        alt_label = cnn_class_names[alt_idx]
                        alt_conf = float(preds[alt_idx])
                        if alt_label in ['plastic', 'metal', 'glass', 'brown-glass', 'green-glass', 'white-glass'] and alt_conf > 0.25:
                            pred_index = alt_idx
                            confidence = alt_conf
                            confidence_margin = confidence - float(preds[top_2_indices[1]] if top_2_indices[1] != alt_idx else preds[top_2_indices[0]])
                            break
            
            if confidence >= 0.40 and confidence_margin >= 0.15:
                all_detections.append({
                    'bbox': bbox,
                    'label': cnn_class_names[pred_index],
                    'confidence': confidence,
                    'source': 'yolo'
                })
    
    # STEP 2: Smart grid detection with texture filtering - COLLECT CROPS
    grid_crops_data = []
    window_sizes = [150, 220]
    stride_ratio = 0.65
    
    for window_size in window_sizes:
        stride = int(window_size * stride_ratio)
        
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                xmin, ymin = x, y
                xmax, ymax = x + window_size, y + window_size
                
                # Skip if already covered by YOLO
                skip = False
                for det in all_detections:
                    if det['source'] == 'yolo':
                        dx1, dy1, dx2, dy2 = det['bbox']
                        ix1, iy1 = max(xmin, dx1), max(ymin, dy1)
                        ix2, iy2 = min(xmax, dx2), min(ymax, dy2)
                        
                        if ix1 < ix2 and iy1 < iy2:
                            overlap = (ix2 - ix1) * (iy2 - iy1)
                            window_area = window_size * window_size
                            if overlap / window_area > 0.4:
                                skip = True
                                break
                
                if skip:
                    continue
                
                crop = image[ymin:ymax, xmin:xmax]
                
                # TEXTURE FILTER
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                std_dev = np.std(gray)
                
                if std_dev < 25:
                    continue
                
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.count_nonzero(edges) / edges.size
                
                if edge_density < 0.05 or edge_density > 0.40:
                    continue
                
                grid_crops_data.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'crop': crop
                })
    
    # BATCH PREDICT for grid crops
    if grid_crops_data:
        batch_crops = np.vstack([preprocess_crop(item['crop']) for item in grid_crops_data])
        batch_preds = cnn_model.predict(batch_crops, verbose=0)
        
        for idx, (preds, item) in enumerate(zip(batch_preds, grid_crops_data)):
            crop = item['crop']
            bbox = item['bbox']
            
            top_2_indices = np.argsort(preds)[-2:][::-1]
            pred_index = top_2_indices[0]
            confidence = float(preds[pred_index])
            second_confidence = float(preds[top_2_indices[1]])
            confidence_margin = confidence - second_confidence
            
            # Fix bias: Penalize shoes/clothes predictions for grid detections
            predicted_label = cnn_class_names[pred_index]
            if predicted_label in ['shoes', 'clothes']:
                gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                texture_std = np.std(gray_check)
                
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hsv_std = np.mean([np.std(hsv_crop[:,:,0]), np.std(hsv_crop[:,:,1]), np.std(hsv_crop[:,:,2])])
                
                if texture_std < 35 or hsv_std < 40:
                    top_3_indices = np.argsort(preds)[-3:][::-1]
                    for alt_idx in top_3_indices[1:]:
                        alt_label = cnn_class_names[alt_idx]
                        alt_conf = float(preds[alt_idx])
                        if alt_label in ['plastic', 'metal', 'glass', 'brown-glass', 'green-glass', 'white-glass'] and alt_conf > 0.40:
                            pred_index = alt_idx
                            confidence = alt_conf
                            confidence_margin = confidence - float(preds[top_2_indices[1]] if top_2_indices[1] != alt_idx else preds[top_2_indices[0]])
                            break
            
            if confidence >= 0.70 and confidence_margin >= 0.25:
                all_detections.append({
                    'bbox': bbox,
                    'label': cnn_class_names[pred_index],
                    'confidence': confidence,
                    'source': 'grid'
                })
    
    # Sort by source (YOLO first) then confidence
    all_detections.sort(key=lambda x: (0 if x['source'] == 'yolo' else 1, -x['confidence']))
    
    # Aggressive NMS to remove overlapping detections
    filtered_detections = []
    for det in all_detections:
        is_duplicate = False
        
        for existing in filtered_detections:
            x1_min, y1_min, x1_max, y1_max = det['bbox']
            x2_min, y2_min, x2_max, y2_max = existing['bbox']
            
            # Intersection
            xi_min = max(x1_min, x2_min)
            yi_min = max(y1_min, y2_min)
            xi_max = min(x1_max, x2_max)
            yi_max = min(y1_max, y2_max)
            
            if xi_min < xi_max and yi_min < yi_max:
                intersection = (xi_max - xi_min) * (yi_max - yi_min)
                area1 = (x1_max - x1_min) * (y1_max - y1_min)
                area2 = (x2_max - x2_min) * (y2_max - y2_min)
                
                # IoU calculation
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                # Also check if one box is mostly contained in another
                overlap_pct1 = intersection / area1 if area1 > 0 else 0
                overlap_pct2 = intersection / area2 if area2 > 0 else 0
                
                # Much stricter overlap detection - mark as duplicate if:
                # 1. High IoU (>20%)
                # 2. One box contains >50% of the other
                # 3. Both boxes are for the same label and overlap >40%
                if iou > 0.20 or overlap_pct1 > 0.50 or overlap_pct2 > 0.50:
                    is_duplicate = True
                    break
                
                # If same label and significant overlap, also mark as duplicate
                if det['label'] == existing['label'] and iou > 0.15:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_detections.append(det)
    
    # Visualize detections
    class_counts = Counter()
    label_positions = []
    total_objects = len(filtered_detections)
    
    for det in filtered_detections:
        xmin, ymin, xmax, ymax = det['bbox']
        label = det['label']
        confidence = det['confidence']
        
        class_counts[label] += 1

        # Extract the object region with small padding
        padding = 3
        xmin_pad = max(0, xmin - padding)
        ymin_pad = max(0, ymin - padding)
        xmax_pad = min(w, xmax + padding)
        ymax_pad = min(h, ymax + padding)
        
        crop = image[ymin_pad:ymax_pad, xmin_pad:xmax_pad].copy()
        
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            # Too small, draw simple rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Create smooth contour around the object
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Heavy blur for smooth edges
            blurred = cv2.GaussianBlur(gray_crop, (15, 15), 0)
            
            # Simple thresholding
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Close gaps and smooth
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest = max(contours, key=cv2.contourArea)
                
                # Heavy smoothing with high epsilon for clean curves
                epsilon = 0.02 * cv2.arcLength(largest, True)  # 2% approximation
                smoothed = cv2.approxPolyDP(largest, epsilon, True)
                
                # Offset to original position
                offset_contour = smoothed + [xmin_pad, ymin_pad]
                
                # Draw smooth contour
                cv2.drawContours(image, [offset_contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Fallback to rounded rectangle
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2, cv2.LINE_AA)
        
        # Prepare text with better formatting
        text = f"{label.upper()}"
        conf_text = f"{confidence * 100:.1f}%"
        
        # Calculate text size with larger font
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        (conf_width, conf_height), conf_baseline = cv2.getTextSize(conf_text, font, font_scale - 0.2, thickness - 1)
        
        # Determine label position - adjust to avoid overlap
        label_y = ymin - 10
        
        # Check for overlap with existing labels and adjust
        for existing_y, existing_xmin, existing_xmax in label_positions:
            if abs(label_y - existing_y) < 40 and not (xmax < existing_xmin or xmin > existing_xmax):
                # Overlap detected, move label higher
                label_y = min(label_y, existing_y - 45)
        
        # Ensure label doesn't go above image
        if label_y - text_height - 35 < 0:
            label_y = ymax + text_height + 35
        
        label_positions.append((label_y, xmin, xmax))
        
        # Draw background rectangle for label with padding
        padding = 8
        bg_y1 = max(0, label_y - text_height - baseline - padding)
        bg_y2 = min(h, label_y + conf_height + padding)
        bg_x2 = min(w, xmin + max(text_width, conf_width) + padding * 2)
        
        # Ensure label coordinates are within bounds
        if bg_y1 < 0 or bg_y2 > h or bg_x2 > w or xmin < 0:
            # Skip label if it would be out of bounds
            continue
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (xmin, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw colored border around label
        cv2.rectangle(image, (xmin, bg_y1), (bg_x2, bg_y2), (0, 255, 0), 2)
        
        # Draw text (class name in white, bold)
        cv2.putText(image, text, (xmin + padding, label_y - baseline - 5),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Draw confidence (in light green, smaller)
        cv2.putText(image, conf_text, (xmin + padding, label_y + conf_height + 5),
                   font, font_scale - 0.2, (144, 238, 144), thickness - 1, cv2.LINE_AA)

    annotated_image_path = os.path.join(ANNOTATED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, image)

    # Waste composition (percentages)
    composition = {}
    if total_objects > 0:
        for k in class_counts:
            composition[k] = round(class_counts[k] / total_objects * 100, 2)
    else:
        # No objects
        composition = {}

    classifications = [{"name": k, "count": v} for k, v in class_counts.items()]
    return annotated_image_path, classifications, composition

def generate_fallback_centers(lat, lng):
    """Generate fallback recycling centers when OSM has no data"""
    # Generate 5 centers in different directions around the location
    import random
    centers = []
    directions = [
        ("North", 0.02, 0),
        ("South", -0.02, 0),
        ("East", 0, 0.02),
        ("West", 0, -0.02),
        ("Central", 0.01, 0.01)
    ]
    
    for direction, lat_offset, lng_offset in directions:
        center_lat = lat + lat_offset + random.uniform(-0.005, 0.005)
        center_lng = lng + lng_offset + random.uniform(-0.005, 0.005)
        
        # Calculate approximate distance
        from math import radians, cos, sin, asin, sqrt
        dlon = radians(center_lng - lng)
        dlat = radians(center_lat - lat)
        a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(center_lat)) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        distance = round(c * 6371, 2)  # km
        
        centers.append({
            "name": f"Waste Management Center - {direction} Area",
            "lat": center_lat,
            "lng": center_lng,
            "distance": distance
        })
    
    centers.sort(key=lambda x: x["distance"])
    return centers

def query_osm_recycling_centers(lat, lng, radius=15000, max_results=5):
    # Very broad search for waste facilities including Indian terms
    query = f"""
    [out:json][timeout:30];
    (
      node(around:{radius},{lat},{lng})["amenity"="recycling"];
      way(around:{radius},{lat},{lng})["amenity"="recycling"];
      node(around:{radius},{lat},{lng})["amenity"="waste_disposal"];
      way(around:{radius},{lat},{lng})["amenity"="waste_disposal"];
      node(around:{radius},{lat},{lng})["amenity"="waste_transfer_station"];
      way(around:{radius},{lat},{lng})["amenity"="waste_transfer_station"];
      node(around:{radius},{lat},{lng})["amenity"="waste_basket"];
      node(around:{radius},{lat},{lng})["landuse"="landfill"];
      way(around:{radius},{lat},{lng})["landuse"="landfill"];
      node(around:{radius},{lat},{lng})["recycling:*"="yes"];
      way(around:{radius},{lat},{lng})["recycling:*"="yes"];
      node(around:{radius},{lat},{lng})["name"~"waste|recycle|garbage|scrap",i];
      way(around:{radius},{lat},{lng})["name"~"waste|recycle|garbage|scrap",i];
    );
    out center;
    """
    url = "http://overpass-api.de/api/interpreter"
    
    try:
        print(f"Searching for recycling centers near ({lat}, {lng}) within {radius}m...")
        response = requests.post(url, data=query, timeout=30)
        
        if response.status_code != 200:
            print(f"OSM API returned status {response.status_code}")
            return generate_fallback_centers(lat, lng)

        data = response.json()
        elements = data.get("elements", [])
        
        print(f"Found {len(elements)} elements from OSM")
        
        if not elements:
            print(f"No facilities found in OSM, using fallback centers")
            return generate_fallback_centers(lat, lng)
        
        centers = []
        for elem in elements:
            tags = elem.get('tags', {})
            if not tags:
                continue
            
            # Get name from various fields
            name = (tags.get('name') or 
                   tags.get('operator') or 
                   tags.get('amenity', '').replace('_', ' ').title() or 
                   'Waste Management Center')
            
            # Get coordinates
            coords = None
            if 'lat' in elem and 'lon' in elem:
                coords = (elem['lat'], elem['lon'])
            elif 'center' in elem:
                coords = (elem['center']['lat'], elem['center']['lon'])
            
            if not coords:
                continue
            
            # Get address if available
            address = tags.get('addr:full') or tags.get('addr:street', '')
            city = tags.get('addr:city', '')
            if address or city:
                location_info = f"{address}, {city}".strip(', ')
                if location_info:
                    name = f"{name} - {location_info}"
            
            centers.append({
                "name": name,
                "lat": coords[0],
                "lng": coords[1]
            })
        
        if not centers:
            print("No valid centers after parsing, using fallback")
            return generate_fallback_centers(lat, lng)
        
        print(f"Successfully parsed {len(centers)} centers")
        
    except Exception as e:
        print(f"Error querying OSM: {e}")
        return generate_fallback_centers(lat, lng)
    
    def haversine(lat1, lon1, lat2, lon2):
        dlon = radians(lon2 - lon1)
        dlat = radians(lat2 - lat1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r
    for center in centers:
        center["distance"] = round(haversine(lat, lng, center["lat"], center["lng"]), 2)
    centers.sort(key=lambda x: x["distance"])
    return centers[:max_results]

# --- API endpoints ---
@app.route('/api/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    try:
        annotated_path, classifications, composition = annotate_image(save_path)
    except Exception as e:
        return jsonify({'error': f'Failed during annotation: {str(e)}'}), 500

    annotated_url = request.host_url + 'static/' + os.path.basename(annotated_path)
    latitude = float(request.form.get('latitude', 0))
    longitude = float(request.form.get('longitude', 0))
    nearest_centers = query_osm_recycling_centers(latitude, longitude)

    return jsonify({
        'annotated_image_url': annotated_url,
        'message': 'Successfully detected and classified objects',
        'classifications': classifications,       # [{name, count}]
        'composition': composition,               # {category: percent, ...}
        'nearest_centers': nearest_centers        # map centers
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
