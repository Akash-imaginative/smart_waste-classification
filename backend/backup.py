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
import time
from functools import lru_cache

# --- App setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# --- Model loading ---
yolo_model = YOLO('yolo11s.pt')
cnn_model = tf.keras.models.load_model("../models/checkpoints/highaccuracy_86.63_epoch29.keras")

cnn_class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# --- Rate limiting for Nominatim API ---
last_nominatim_request = 0
NOMINATIM_RATE_LIMIT = 1.0  # 1 second between requests (Nominatim requirement)
nominatim_cache = {}  # Cache for coordinates

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
        # Lower confidence for more detections
        results = yolo_model(rgb_image, conf=0.30, iou=0.50, max_det=100, verbose=False)
        if results and results[0].boxes:
            yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in yolo_boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                
                if (xmax - xmin) < 20 or (ymax - ymin) < 20:  # Lowered from 30
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
            
            # STRICT RULE: Battery, clothes, and shoes require 100% confidence, else classify as trash
            final_label = cnn_class_names[pred_index]
            if final_label in ['battery', 'clothes', 'shoes']:
                if confidence < 0.995:  # 99.5% threshold (effectively 100%)
                    final_label = 'trash'
                    confidence = 0.85  # High confidence for trash classification
            
            if confidence >= 0.55 and confidence_margin >= 0.30:
                all_detections.append({
                    'bbox': bbox,
                    'label': final_label,
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
            
            # STRICT RULE: Battery, clothes, and shoes require 100% confidence, else classify as trash
            final_label = cnn_class_names[pred_index]
            if final_label in ['battery', 'clothes', 'shoes']:
                if confidence < 0.995:  # 99.5% threshold (effectively 100%)
                    final_label = 'trash'
                    confidence = 0.90  # High confidence for trash classification
            
            if confidence >= 0.70 and confidence_margin >= 0.25:
                all_detections.append({
                    'bbox': bbox,
                    'label': final_label,
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
            # Too small, draw rounded rectangle
            radius = min(15, (xmax-xmin)//4, (ymax-ymin)//4)
            cv2.rectangle(image, (xmin+radius, ymin), (xmax-radius, ymax), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.ellipse(image, (xmin+radius, ymin+radius), (radius, radius), 180, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.ellipse(image, (xmax-radius, ymin+radius), (radius, radius), 270, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.ellipse(image, (xmax-radius, ymax-radius), (radius, radius), 0, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.ellipse(image, (xmin+radius, ymax-radius), (radius, radius), 90, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            # Create ultra-smooth organic contour
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Extreme smoothing pipeline
            blurred = cv2.GaussianBlur(gray_crop, (25, 25), 0)
            blurred = cv2.bilateralFilter(blurred, 11, 100, 100)
            blurred = cv2.GaussianBlur(blurred, (15, 15), 0)
            
            # Adaptive threshold with larger block
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 15, 3)
            
            # Extreme morphological smoothing
            kernel_xl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            kernel_md = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            
            # Multiple passes for ultra-smooth boundaries
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_xl, iterations=3)
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_lg, iterations=2)
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_md, iterations=2)
            
            # Final blur pass on mask
            closed = cv2.GaussianBlur(closed, (15, 15), 0)
            _, closed = cv2.threshold(closed, 150, 255, cv2.THRESH_BINARY)
            
            # Find contours with maximum smoothing
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                
                # BEST SMOOTHING: Convex hull first
                hull = cv2.convexHull(largest)
                
                # Downsample points for smoother curves
                epsilon = 0.008 * cv2.arcLength(hull, True)  # 0.8% for ultra-smooth
                smoothed = cv2.approxPolyDP(hull, epsilon, True)
                
                # Try ellipse fitting for organic curves
                final_contour = smoothed
                if len(smoothed) >= 5:
                    try:
                        # Fit ellipse and use it
                        ellipse = cv2.fitEllipse(smoothed)
                        # Create ellipse mask and extract contour
                        temp_mask = np.zeros((crop.shape[0]+20, crop.shape[1]+20), dtype=np.uint8)
                        cv2.ellipse(temp_mask, 
                                  (ellipse[0][0]+10, ellipse[0][1]+10),
                                  (int(ellipse[1][0]/2), int(ellipse[1][1]/2)),
                                  ellipse[2], 0, 360, 255, -1)
                        # Smooth the ellipse
                        temp_mask = cv2.GaussianBlur(temp_mask, (11, 11), 0)
                        _, temp_mask = cv2.threshold(temp_mask, 200, 255, cv2.THRESH_BINARY)
                        ell_cnt, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if ell_cnt and len(ell_cnt[0]) > 50:
                            # Subsample for performance
                            final_contour = ell_cnt[0][::3] - [10, 10]
                    except:
                        final_contour = smoothed
                
                # Offset to original position
                offset_contour = final_contour + [xmin_pad, ymin_pad]
                
                # Draw with anti-aliasing and thicker line
                cv2.drawContours(image, [offset_contour], -1, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                # Fallback: perfect rounded rectangle
                radius = min(20, (xmax-xmin)//4, (ymax-ymin)//4)
                cv2.rectangle(image, (xmin+radius, ymin), (xmax-radius, ymax), (0, 255, 0), 3, cv2.LINE_AA)
                cv2.ellipse(image, (xmin+radius, ymin+radius), (radius, radius), 180, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.ellipse(image, (xmax-radius, ymin+radius), (radius, radius), 270, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.ellipse(image, (xmax-radius, ymax-radius), (radius, radius), 0, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.ellipse(image, (xmin+radius, ymax-radius), (radius, radius), 90, 0, 90, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Prepare text with better formatting
        text = f"{label.upper()}"
        conf_text = f"{confidence * 100:.1f}%"
        
        # Calculate text size with larger font
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        (conf_width, conf_height), conf_baseline = cv2.getTextSize(conf_text, font, font_scale - 0.2, thickness - 1)
        
        padding = 8
        max_label_width = max(text_width, conf_width) + padding * 2
        total_label_height = text_height + conf_height + baseline + padding * 2
        
        # Smart label positioning with boundary checks
        # Try position 1: Above the box
        label_x = xmin
        label_y = ymin - total_label_height - 5
        
        # Adjust X if label goes beyond right edge
        if label_x + max_label_width > w:
            label_x = max(0, w - max_label_width - 5)
        
        # If label goes above image, place it below the box
        if label_y < 0:
            label_y = ymax + 5
            # If still doesn't fit below, place inside the box at top
            if label_y + total_label_height > h:
                label_y = ymin + 5
        
        # Final bounds check - ensure everything fits
        label_x = max(5, min(label_x, w - max_label_width - 5))
        label_y = max(total_label_height, min(label_y, h - 10))
        
        # Check for overlap with existing labels
        for existing_y, existing_xmin, existing_xmax in label_positions:
            if abs(label_y - existing_y) < total_label_height + 5:
                x_overlap = not (label_x + max_label_width < existing_xmin or label_x > existing_xmax)
                if x_overlap:
                    # Move label down to avoid overlap
                    label_y = existing_y + total_label_height + 5
                    # If moved beyond image, place inside box
                    if label_y + total_label_height > h:
                        label_y = max(total_label_height, ymin + total_label_height + 5)
        
        label_positions.append((label_y, label_x, label_x + max_label_width))
        
        # Calculate final label rectangle coordinates
        bg_x1 = int(label_x)
        bg_y1 = int(label_y - total_label_height + padding)
        bg_x2 = int(label_x + max_label_width)
        bg_y2 = int(label_y + padding)
        
        # Final safety clamp to image boundaries
        bg_x1 = max(0, min(bg_x1, w - 10))
        bg_y1 = max(0, min(bg_y1, h - 10))
        bg_x2 = max(bg_x1 + 10, min(bg_x2, w))
        bg_y2 = max(bg_y1 + 10, min(bg_y2, h))
        
        # Skip if label area is invalid
        if bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
            continue
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw colored border around label
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), 2)
        
        # Calculate text positions within the label box
        text_x = bg_x1 + padding
        text_y = bg_y1 + text_height + baseline + padding
        conf_y = text_y + conf_height + 5
        
        # Ensure text stays within label bounds
        text_x = max(bg_x1 + 2, min(text_x, bg_x2 - text_width - 2))
        text_y = max(bg_y1 + text_height + 2, min(text_y, bg_y2 - conf_height - 5))
        conf_y = max(text_y + 5, min(conf_y, bg_y2 - 2))
        
        # Draw text (class name in white, bold)
        cv2.putText(image, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Draw confidence (in light green, smaller)
        cv2.putText(image, conf_text, (text_x, conf_y),
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
        
        # Calculate approximate distance with safe math
        from math import radians, cos, sin, asin, sqrt
        dlon = radians(center_lng - lng)
        dlat = radians(center_lat - lat)
        a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(center_lat)) * sin(dlon/2)**2
        c = 2 * asin(sqrt(min(1.0, a)))  # Clamp to avoid domain error
        distance = round(c * 6371, 2)  # km
        
        centers.append({
            "name": f"Waste Management Center - {direction} Area",
            "lat": center_lat,
            "lng": center_lng,
            "distance": distance,
            "full_address": f"General {direction} Area (Approximate location)"
        })
    
    centers.sort(key=lambda x: x["distance"])
    return centers

def query_osm_recycling_centers(lat, lng, radius=15000, max_results=5):
    """Get recycling centers WITH real neighborhood names via Nominatim"""
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lng})["amenity"="recycling"];
      way(around:{radius},{lat},{lng})["amenity"="recycling"];
      node(around:{radius},{lat},{lng})["amenity"="waste_disposal"];
      way(around:{radius},{lat},{lng})["amenity"="waste_disposal"];
      node(around:{radius},{lat},{lng})["amenity"="waste_transfer_station"];
    );
    out center;
    """
    
    url = "http://overpass-api.de/api/interpreter"
    
    try:
        print(f"🔍 Searching OSM near ({lat:.4f}, {lng:.4f})...")
        response = requests.post(url, data=query, timeout=25)
        
        if response.status_code != 200:
            return generate_fallback_centers(lat, lng)

        data = response.json()
        elements = data.get("elements", [])
        
        if not elements:
            return generate_fallback_centers(lat, lng)
        
        centers = []
        # Blacklist of closed/invalid shops
        blacklisted_names = [
            'mahesh paper mart',
            'mahesh paper',
            'paper mart'
        ]
        
        for elem in elements:
            tags = elem.get('tags', {})
            if not tags:
                continue
            
            # Base name
            base_name = (tags.get('name') or 
                        tags.get('operator') or 
                        tags.get('amenity', 'Recycling').replace('_', ' ').title())
            
            # Filter out blacklisted names
            if any(blacklisted.lower() in base_name.lower() for blacklisted in blacklisted_names):
                print(f"🚫 Filtered out blacklisted center: {base_name}")
                continue
            
            # Coordinates
            if 'lat' in elem and 'lon' in elem:
                center_lat, center_lng = elem['lat'], elem['lon']
            elif 'center' in elem:
                center_lat, center_lng = elem['center']['lat'], elem['center']['lon']
            else:
                continue
            
            # 🔥 IMPROVED NOMINATIM: Rate-limited, cached, with retry logic
            area_name = 'Local Area'
            full_address = 'Address not available'
            
            try:
                # Declare global variables at the start
                global nominatim_cache, last_nominatim_request
                
                # Check cache first
                cache_key = f"{center_lat:.6f},{center_lng:.6f}"
                if cache_key in nominatim_cache:
                    cached = nominatim_cache[cache_key]
                    area_name = cached['area_name']
                    full_address = cached['full_address']
                    print(f"📦 Using cached address for {cache_key}")
                else:
                    # Rate limiting: wait if needed
                    time_since_last = time.time() - last_nominatim_request
                    if time_since_last < NOMINATIM_RATE_LIMIT:
                        sleep_time = NOMINATIM_RATE_LIMIT - time_since_last
                        print(f"⏳ Rate limiting: sleeping {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                    
                    nominatim_url = "https://nominatim.openstreetmap.org/reverse"
                    nominatim_headers = {
                        'User-Agent': 'WasteClassificationApp/1.0 (educational-project)'
                    }
                    nominatim_params = {
                        'lat': center_lat,
                        'lon': center_lng,
                        'format': 'json',
                        'addressdetails': 1,
                        'zoom': 18,  # Higher zoom for precise neighborhood
                        'accept-language': 'en'
                    }
                    
                    # Retry logic with exponential backoff
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            last_nominatim_request = time.time()
                            nom_resp = requests.get(
                                nominatim_url, 
                                headers=nominatim_headers, 
                                params=nominatim_params, 
                                timeout=10
                            )
                            
                            if nom_resp.status_code == 200:
                                area_data = nom_resp.json()
                                full_address = area_data.get('display_name', 'Address not available')
                                
                                addr = area_data.get('address', {})
                                
                                # 🎯 IMPROVED: More comprehensive location extraction
                                area_parts = []
                                
                                # Try multiple location fields in priority order
                                location_fields = [
                                    'neighbourhood', 'suburb', 'quarter', 'city_district',
                                    'district', 'borough', 'road', 'hamlet', 'village',
                                    'town', 'city', 'municipality'
                                ]
                                
                                for field in location_fields:
                                    if addr.get(field):
                                        area_parts.append(addr[field])
                                        break  # Use first available
                                
                                area_name = area_parts[0] if area_parts else 'Local Area'
                                
                                # Cache the result
                                nominatim_cache[cache_key] = {
                                    'area_name': area_name,
                                    'full_address': full_address
                                }
                                print(f"✅ Nominatim success for {cache_key}: {area_name}")
                                break  # Success, exit retry loop
                                
                            elif nom_resp.status_code == 429:  # Too Many Requests
                                wait_time = 2 ** attempt  # Exponential backoff
                                print(f"⚠ Rate limited (429), waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                                time.sleep(wait_time)
                            else:
                                print(f"⚠ Nominatim returned {nom_resp.status_code}, attempt {attempt+1}/{max_retries}")
                                if attempt < max_retries - 1:
                                    time.sleep(1)
                                    
                        except requests.exceptions.Timeout:
                            print(f"⏱ Nominatim timeout, attempt {attempt+1}/{max_retries}")
                            if attempt < max_retries - 1:
                                time.sleep(1)
                        except requests.exceptions.RequestException as req_err:
                            print(f"🌐 Nominatim request error: {req_err}, attempt {attempt+1}/{max_retries}")
                            if attempt < max_retries - 1:
                                time.sleep(1)
                                
            except Exception as nom_err:
                print(f"❌ Nominatim outer error: {nom_err}")
                area_name = 'Local Area'
                full_address = 'Address not available'
            
            # Final name
            full_name = f"{base_name} - {area_name}"
            print(f"✅ Center: {full_name}")
            
            centers.append({
                "name": full_name,
                "full_address": full_address,
                "lat": center_lat,
                "lng": center_lng
            })
        
        # Calculate distances with improved accuracy
        def haversine(lat1, lon1, lat2, lon2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            Returns distance in kilometers with high precision
            """
            from math import radians, cos, sin, asin, sqrt, atan2
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula with improved precision
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            
            # Clamp 'a' to avoid floating-point errors with asin
            a = max(0, min(1, a))
            
            c = 2 * atan2(sqrt(a), sqrt(1-a))  # More accurate than asin
            
            # Earth's radius in km (mean radius)
            R = 6371.0088
            
            return R * c
        
        # Calculate distances with 3 decimal precision for accuracy
        for center in centers:
            distance_km = haversine(lat, lng, center["lat"], center["lng"])
            center["distance"] = round(distance_km, 3)  # 3 decimal places for meter precision
        
        seen_coords = set()
        unique_centers = []
        for center in centers:
            coord_key = f"{center['lat']:.6f},{center['lng']:.6f}"
            if coord_key not in seen_coords:
                seen_coords.add(coord_key)
                unique_centers.append(center)
        
        unique_centers.sort(key=lambda x: x["distance"])
        return unique_centers[:max_results]
        
    except Exception as e:
        print(f"❌ OSM Error: {e}")
        return generate_fallback_centers(lat, lng)


# --- API endpoints ---
@app.route('/api/find-centers', methods=['POST'])
def find_centers():
    """Find recycling centers without requiring image upload"""
    try:
        data = request.get_json()
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        
        if latitude == 0 or longitude == 0:
            return jsonify({'error': 'Valid latitude and longitude required'}), 400
        
        nearest_centers = query_osm_recycling_centers(latitude, longitude)
        
        return jsonify({
            'nearest_centers': nearest_centers,
            'message': 'Successfully found recycling centers'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to find centers: {str(e)}'}), 500

@app.route('/api/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Validate image file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
    
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