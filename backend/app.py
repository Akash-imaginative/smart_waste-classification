from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from collections import Counter
import requests
from math import radians, cos, sin, asin, sqrt

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

yolo_model = YOLO('yolov5s.pt')
cnn_model = load_model('C:/Users/Angelina/Desktop/Waste-Classification/models/densenet121_waste_classifier.keras')

cnn_class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

def preprocess_crop(crop_img):
    crop_img = cv2.resize(crop_img, (224, 224))
    crop_img = crop_img.astype("float") / 255.0
    crop_img = img_to_array(crop_img)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img

def annotate_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo_model(rgb_image)
    detections = results[0].boxes.xyxy.cpu().numpy()

    class_counts = Counter()

    for box in detections:
        xmin, ymin, xmax, ymax = map(int, box)
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue  # Skip empty crops

        processed_crop = preprocess_crop(crop)
        preds = cnn_model.predict(processed_crop)
        pred_index = np.argmax(preds[0])
        label = cnn_class_names[pred_index]
        confidence = preds[0][pred_index]

        class_counts[label] += 1

        # Draw bounding box and label
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        text = f"{label}: {confidence * 100:.1f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            image,
            (xmin, ymin - text_height - baseline),
            (xmin + text_width, ymin),
            (0, 0, 0), -1
        )
        cv2.putText(
            image, text, (xmin, ymin - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2
        )

    annotated_image_path = os.path.join(ANNOTATED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, image)

    classifications = [{"name": k, "count": v} for k, v in class_counts.items()]

    return annotated_image_path, classifications

def query_osm_recycling_centers(lat, lng, radius=5000, max_results=3):
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lng})["amenity"="recycling"];
      way(around:{radius},{lat},{lng})["amenity"="recycling"];
      relation(around:{radius},{lat},{lng})["amenity"="recycling"];
    );
    out center;
    """


    url = "http://overpass-api.de/api/interpreter"
    response = requests.post(url, data=query)

    if response.status_code != 200:
        return []

    data = response.json()
    elements = data.get("elements", [])

    centers = []
    for elem in elements:
        if 'tags' not in elem:
            continue
        name = elem['tags'].get('name', 'Unnamed Center')
        if 'lat' in elem and 'lon' in elem:
            coords = (elem['lat'], elem['lon'])
        elif 'center' in elem:
            coords = (elem['center']['lat'], elem['center']['lon'])
        else:
            continue

        centers.append({
            "name": name,
            "lat": coords[0],
            "lng": coords[1],
        })

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
        annotated_path, classifications = annotate_image(save_path)
    except Exception as e:
        return jsonify({'error': f'Failed during annotation: {str(e)}'}), 500

    annotated_url = request.host_url + 'static/' + os.path.basename(annotated_path)

    latitude = float(request.form.get('latitude', 0))
    longitude = float(request.form.get('longitude', 0))

    nearest_centers = query_osm_recycling_centers(latitude, longitude)

    return jsonify({
        'annotated_image_url': annotated_url,
        'message': 'Successfully detected and classified objects',
        'classifications': classifications,
        'nearest_centers': nearest_centers
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
