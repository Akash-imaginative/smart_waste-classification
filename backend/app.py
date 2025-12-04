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
yolo_model = YOLO('yolov5s.pt')
# Use the best model from evaluation (epoch 29 with 80.25% accuracy)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'highaccuracy_86.63_epoch29.keras')
cnn_model = tf.keras.models.load_model(model_path)

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
    if image is None or image.size == 0:
        raise ValueError(f"Failed to load image from {image_path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    all_detections = []
    
    # BATCH OPTIMIZATION: Collect all crops first, then predict in one batch
    yolo_crops_data = []  # Store (crop, bbox, crop_image) for YOLO detections
    
    # STEP 1: YOLO for distinct objects - COLLECT CROPS
    try:
        # First pass: global detection with permissive thresholds to find more objects
        results = yolo_model(rgb_image, conf=0.08, iou=0.45, max_det=300, verbose=False)
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

        # Second pass: tiled inference to catch small/edge objects
        tile_size = 640
        stride = int(tile_size * 0.75)  # overlap tiles
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(h, y0 + tile_size)
                x1 = min(w, x0 + tile_size)
                tile = rgb_image[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                t_results = yolo_model(tile, conf=0.08, iou=0.50, max_det=150, verbose=False)
                if t_results and t_results[0].boxes:
                    t_boxes = t_results[0].boxes.xyxy.cpu().numpy()
                    for box in t_boxes:
                        txmin, tymin, txmax, tymax = map(int, box)
                        xmin, ymin = max(0, x0 + txmin), max(0, y0 + tymin)
                        xmax, ymax = min(w, x0 + txmax), min(h, y0 + tymax)
                        if (xmax - xmin) < 24 or (ymax - ymin) < 24:
                            continue
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
            
            # Fix bias: Penalize problematic class predictions based on evaluation results
            predicted_label = cnn_class_names[pred_index]
            
            # Clothes: 0% F1-score in evaluation - be very skeptical
            if predicted_label == 'clothes':
                gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                texture_std = np.std(gray_check)
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hsv_std = np.mean([np.std(hsv_crop[:,:,0]), np.std(hsv_crop[:,:,1]), np.std(hsv_crop[:,:,2])])
                
                if confidence < 0.95 or texture_std < 40 or hsv_std < 45:
                    top_5_indices = np.argsort(preds)[-5:][::-1]
                    for alt_idx in top_5_indices[1:]:
                        alt_label = cnn_class_names[alt_idx]
                        if alt_label != 'clothes' and preds[alt_idx] > 0.20:
                            pred_index = alt_idx
                            confidence = float(preds[alt_idx])
                            break
            
            # Shoes: needs strong evidence
            elif predicted_label == 'shoes':
                gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                texture_std = np.std(gray_check)
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hsv_std = np.mean([np.std(hsv_crop[:,:,0]), np.std(hsv_crop[:,:,1]), np.std(hsv_crop[:,:,2])])
                
                if texture_std < 35 or hsv_std < 40:
                    top_3_indices = np.argsort(preds)[-3:][::-1]
                    for alt_idx in top_3_indices[1:]:
                        alt_label = cnn_class_names[alt_idx]
                        if alt_label in ['plastic', 'metal', 'glass', 'brown-glass', 'green-glass', 'white-glass', 'trash'] and preds[alt_idx] > 0.30:
                            pred_index = alt_idx
                            confidence = float(preds[alt_idx])
                            break
            
            # Final threshold check
            final_label = cnn_class_names[pred_index]
            if final_label == 'clothes' and confidence < 0.95:
                final_label = 'trash'
            elif final_label in ['shoes', 'metal'] and confidence < 0.85:
                final_label = 'trash'
            
            if confidence >= 0.40 and confidence_margin >= 0.15:
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
            
            # Fix bias: HEAVILY penalize clothes predictions (0% F1-score in evaluation)
            predicted_label = cnn_class_names[pred_index]
            
            if predicted_label == 'clothes':
                # Clothes had 100% misclassification rate - be extremely skeptical
                gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                texture_std = np.std(gray_check)
                
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hsv_std = np.mean([np.std(hsv_crop[:,:,0]), np.std(hsv_crop[:,:,1]), np.std(hsv_crop[:,:,2])])
                
                # Unless it has very strong fabric-like characteristics AND high confidence, switch to alternative
                if confidence < 0.95 or texture_std < 40 or hsv_std < 45:
                    top_5_indices = np.argsort(preds)[-5:][::-1]
                    for alt_idx in top_5_indices[1:]:
                        alt_label = cnn_class_names[alt_idx]
                        alt_conf = float(preds[alt_idx])
                        if alt_label != 'clothes' and alt_conf > 0.20:
                            pred_index = alt_idx
                            confidence = alt_conf
                            confidence_margin = confidence - float(preds[top_5_indices[1]] if top_5_indices[1] != alt_idx else preds[top_5_indices[0]])
                            break
            
            elif predicted_label == 'shoes':
                # Similar logic for shoes
                gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                texture_std = np.std(gray_check)
                
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hsv_std = np.mean([np.std(hsv_crop[:,:,0]), np.std(hsv_crop[:,:,1]), np.std(hsv_crop[:,:,2])])
                
                if texture_std < 35 or hsv_std < 40:
                    top_3_indices = np.argsort(preds)[-3:][::-1]
                    for alt_idx in top_3_indices[1:]:
                        alt_label = cnn_class_names[alt_idx]
                        alt_conf = float(preds[alt_idx])
                        if alt_label in ['plastic', 'metal', 'glass', 'brown-glass', 'green-glass', 'white-glass', 'trash'] and alt_conf > 0.35:
                            pred_index = alt_idx
                            confidence = alt_conf
                            confidence_margin = confidence - float(preds[top_2_indices[1]] if top_2_indices[1] != alt_idx else preds[top_2_indices[0]])
                            break
            
            # Apply VERY strict accuracy threshold for problematic classes
            final_label = cnn_class_names[pred_index]
            
            # Based on evaluation: clothes had 0% F1-score, always confused with trash
            # Paper had 3.92% F1-score, confused with cardboard and trash
            if final_label == 'clothes':
                if confidence < 0.95 or confidence_margin < 0.35:  # 95% confidence + strong margin
                    # Check if it's actually trash or other material
                    top_5_indices = np.argsort(preds)[-5:][::-1]
                    for alt_idx in top_5_indices[1:]:
                        alt_label = cnn_class_names[alt_idx]
                        if alt_label != 'clothes' and preds[alt_idx] > 0.15:
                            final_label = alt_label
                            confidence = float(preds[alt_idx])
                            break
                    else:
                        final_label = 'trash'  # Default to trash if no good alternative
            
            elif final_label in ['shoes', 'metal']:
                if confidence < 0.85:  # 85% threshold for shoes/metal
                    final_label = 'trash'  # Reclassify as trash if below threshold
            
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
        
        padding = 8
        max_label_width = max(text_width, conf_width) + padding * 2
        total_label_height = text_height + conf_height + baseline + padding * 2
        
        # Helper function to check if two rectangles overlap
        def rectangles_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
            return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
        
        # Try multiple positions to avoid overlaps
        positions_to_try = [
            (xmin, ymin - total_label_height - 10, 'above'),           # Above box
            (xmin, ymax + 5, 'below'),                                  # Below box
            (xmax + 5, ymin, 'right'),                                  # Right of box
            (max(0, xmin - max_label_width - 5), ymin, 'left'),        # Left of box
            (xmin, ymin + 5, 'inside_top'),                            # Inside box top
            (xmin, ymax - total_label_height - 5, 'inside_bottom'),    # Inside box bottom
        ]
        
        label_x, label_y = xmin, ymin - total_label_height - 10
        best_position = None
        
        for try_x, try_y, position_type in positions_to_try:
            # Calculate candidate label bounds
            candidate_y1 = try_y
            candidate_y2 = try_y + total_label_height
            candidate_x1 = try_x
            candidate_x2 = try_x + max_label_width
            
            # Ensure within image bounds
            if candidate_y1 < 0 or candidate_y2 > h or candidate_x1 < 0 or candidate_x2 > w:
                continue
            
            # Check for overlaps with existing labels
            has_overlap = False
            for (ex1, ey1, ex2, ey2) in label_positions:
                if rectangles_overlap(candidate_x1, candidate_y1, max_label_width, total_label_height,
                                     ex1, ey1, ex2 - ex1, ey2 - ey1):
                    has_overlap = True
                    break
            
            # Found a position without overlap
            if not has_overlap:
                label_x = try_x
                label_y = try_y + text_height + baseline + padding  # Adjust for text baseline
                best_position = position_type
                break
        
        # If no non-overlapping position found, use original with slight offset
        if best_position is None:
            label_x = xmin
            label_y = ymin - 10
            # Shift down if overlaps exist
            for (ex1, ey1, ex2, ey2) in label_positions:
                if rectangles_overlap(xmin, ymin - total_label_height - 10, max_label_width, total_label_height,
                                     ex1, ey1, ex2 - ex1, ey2 - ey1):
                    label_y = ey2 + text_height + 5
        
        # Final boundary check
        if label_x + max_label_width > w:
            label_x = max(0, w - max_label_width - 5)
        if label_y - text_height - baseline - padding < 0:
            label_y = text_height + baseline + padding + 5
        if label_y + conf_height + padding > h:
            label_y = h - conf_height - padding - 5
        
        # Calculate background rectangle bounds
        bg_y1 = label_y - text_height - baseline - padding
        bg_y2 = label_y + conf_height + padding
        bg_x1 = label_x
        bg_x2 = label_x + max_label_width
        
        # Store label position for future collision detection (before clamping)
        label_positions.append((bg_x1, bg_y1, bg_x2, bg_y2))
        
        # Clamp to image boundaries
        bg_y1 = max(0, bg_y1)
        bg_y2 = min(h, bg_y2)
        bg_x1 = max(0, bg_x1)
        bg_x2 = min(w, bg_x2)
        
        # Ensure label coordinates are within bounds and valid
        if bg_y2 <= bg_y1 or bg_x2 <= bg_x1:
            continue
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw colored border around label
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), 2)
        
        # Draw text (class name in white, bold) - ensure text position is within bounds
        text_x = max(bg_x1 + padding, 0)
        text_y = max(label_y - baseline - 5, text_height)
        cv2.putText(image, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Draw confidence (in light green, smaller)
        conf_y = min(label_y + conf_height + 5, h - 5)
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
                # Check cache first
                cache_key = f"{center_lat:.6f},{center_lng:.6f}"
                if cache_key in nominatim_cache:
                    cached = nominatim_cache[cache_key]
                    area_name = cached['area_name']
                    full_address = cached['full_address']
                    print(f"📦 Using cached address for {cache_key}")
                else:
                    # Rate limiting: wait if needed
                    global last_nominatim_request
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
                                print(f"⚠️ Rate limited (429), waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                                time.sleep(wait_time)
                            else:
                                print(f"⚠️ Nominatim returned {nom_resp.status_code}, attempt {attempt+1}/{max_retries}")
                                if attempt < max_retries - 1:
                                    time.sleep(1)
                                    
                        except requests.exceptions.Timeout:
                            print(f"⏱️ Nominatim timeout, attempt {attempt+1}/{max_retries}")
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
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
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
