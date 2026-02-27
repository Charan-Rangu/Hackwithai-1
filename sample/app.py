import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load BLIP models from vision_llm directly
from vision_llm import analyze_cleanliness

app = Flask(__name__)
CORS(app)

# Load YOLO models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ppe_model = YOLO(os.path.join(BASE_DIR, "foodsafety.pt"))
kitchen_model = YOLO(os.path.join(BASE_DIR, "kitchenobj.pt"))

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    # Save the original image
    image_path = os.path.join(BASE_DIR, "temp_image.jpg")
    file.save(image_path)
    
    # Analyze with vision_llm (BLIP)
    caption, hygiene_score = analyze_cleanliness(image_path)
    
    # Analyze with YOLO
    img_cv2 = cv2.imread(image_path)
    h_img, w_img, _ = img_cv2.shape
    
    ppe_results = ppe_model(image_path, conf=0.05)
    kitchen_results = kitchen_model(image_path, conf=0.25)
    
    detected_ppe = []
    violations = []
    detections = []
    
    person_detected = any(ppe_results[0].names[int(box.cls)].lower() == 'person' for box in ppe_results[0].boxes)
    cap_detected = any(ppe_results[0].names[int(box.cls)].lower() == 'medical_cap' for box in ppe_results[0].boxes)
    
    if person_detected and not cap_detected:
        violations.append("Missing PPE: medical cap")
        hygiene_score -= 10
        
    for box in ppe_results[0].boxes:
        conf = float(box.conf)
        cls_id = int(box.cls)
        label = ppe_results[0].names[cls_id]
        
        if label.lower() not in ['gloves', 'no_gloves'] and conf < 0.25:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        
        x_norm = x1 / w_img
        y_norm = y1 / h_img
        w_norm = (x2 - x1) / w_img
        h_norm = (y2 - y1) / h_img
        
        if label.lower() in ['no_gloves', 'without_mask', 'mask_weared_incorrect', 'no_apron'] and not person_detected:
            continue
            
        status = 'ok'
        if label.lower() in ['no_gloves', 'without_mask', 'mask_weared_incorrect', 'no_apron']:
            status = 'bad'
            violations.append(f"Missing PPE: {label.replace('_', ' ')}")
        elif label.lower() in ['gloves', 'mask', 'medical_cap', 'apron', 'with_mask', 'person']:
            detected_ppe.append(label.replace('_', ' '))
            
        detections.append({
            'label': label,
            'x': int(x_norm * 500),
            'y': int(y_norm * 300),
            'w': int(w_norm * 500),
            'h': int(h_norm * 300),
            'status': status
        })
        
    for box in kitchen_results[0].boxes:
        conf = float(box.conf)
        cls_id = int(box.cls)
        label = kitchen_results[0].names[cls_id]
        
        if label.lower() not in ['gloves', 'no_gloves'] and conf < 0.30:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        x_norm = x1 / w_img
        y_norm = y1 / h_img
        w_norm = (x2 - x1) / w_img
        h_norm = (y2 - y1) / h_img
        
        if label.lower() in ['no_apron', 'without_mask', 'mask_weared_incorrect'] and not person_detected:
            continue
            
        status = 'ok'
        if label.lower() in ['fire', 'smoke', 'no_apron', 'without_mask', 'mask_weared_incorrect']:
            status = 'bad'
            violations.append(f"Hazard: {label.replace('_', ' ')}")
            hygiene_score -= 10
        elif label.lower() in ['apron', 'with_mask']:
            detected_ppe.append(label.replace('_', ' '))
            
        detections.append({
            'label': label.replace('_', ' '),
            'x': int(x_norm * 500),
            'y': int(y_norm * 300),
            'w': int(w_norm * 500),
            'h': int(h_norm * 300),
            'status': status
        })
        
    hygiene_score = max(0, min(100, int(hygiene_score)))
    
    # Deduplicate
    detected_ppe = list(set(detected_ppe))
    violations = list(set(violations))
    return jsonify({
        "compliance_score": hygiene_score,
        "detected_ppe": detected_ppe,
        "violations": violations,
        "detections": detections,
        "caption": caption
    })

OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

from flask import send_from_directory

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video part"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected video"}), 400
    
    input_video_path = os.path.join(BASE_DIR, "temp_video.mp4")
    file.save(input_video_path)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 500
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30
    
    output_filename = "processed_video.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    incidents = 0
    frequent_violations = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        ppe_results = ppe_model(frame, conf=0.05)
        kitchen_results = kitchen_model(frame, conf=0.25)
        
        frame_has_incident = False
        
        person_detected = any(ppe_results[0].names[int(box.cls)].lower() == 'person' for box in ppe_results[0].boxes)
        cap_detected = any(ppe_results[0].names[int(box.cls)].lower() == 'medical_cap' for box in ppe_results[0].boxes)
        
        if person_detected and not cap_detected:
            frame_has_incident = True
            if frame_count % int(fps) == 0:
                mins, secs = divmod(frame_count // int(fps), 60)
                v_str = f"T-{mins:02d}:{secs:02d} - Missing PPE: medical cap"
                if v_str not in frequent_violations: frequent_violations.append(v_str)
                
        for box in ppe_results[0].boxes:
            conf = float(box.conf)
            label = ppe_results[0].names[int(box.cls)]
            
            if label.lower() not in ['gloves', 'no_gloves'] and conf < 0.25:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if label.lower() in ['no_gloves', 'without_mask', 'mask_weared_incorrect', 'no_apron'] and not person_detected:
                continue
                
            status_color = (0, 255, 0)
            if label.lower() in ['no_gloves', 'without_mask', 'mask_weared_incorrect', 'no_apron']:
                status_color = (0, 0, 255)
                frame_has_incident = True
                if frame_count % int(fps) == 0:
                    mins, secs = divmod(frame_count // int(fps), 60)
                    v_str = f"T-{mins:02d}:{secs:02d} - Missing PPE: {label.replace('_', ' ')}"
                    if v_str not in frequent_violations: frequent_violations.append(v_str)
                    
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
            cv2.putText(frame, f"{label.replace('_', ' ')}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
        for box in kitchen_results[0].boxes:
            conf = float(box.conf)
            label = kitchen_results[0].names[int(box.cls)]
            
            if label.lower() not in ['gloves', 'no_gloves'] and conf < 0.30:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if label.lower() in ['no_apron', 'without_mask', 'mask_weared_incorrect'] and not person_detected:
                continue
                
            status_color = (0, 255, 0)
            if label.lower() in ['fire', 'smoke', 'no_apron', 'without_mask', 'mask_weared_incorrect']:
                status_color = (0, 0, 255)
                frame_has_incident = True
                if frame_count % int(fps) == 0:
                    mins, secs = divmod(frame_count // int(fps), 60)
                    v_str = f"T-{mins:02d}:{secs:02d} - Hazard: {label.replace('_', ' ')}"
                    if v_str not in frequent_violations: frequent_violations.append(v_str)
                    
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
            cv2.putText(frame, f"{label.replace('_', ' ')}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
        if frame_has_incident: incidents += 1
        out.write(frame)
        
    cap.release()
    out.release()
    
    score = max(0, 100 - (incidents // int(fps)) * 2)
    
    return jsonify({
        "average_score": score,
        "frames_analyzed": frame_count,
        "critical_incidents": len(frequent_violations),
        "frequent_violations": frequent_violations[:10],
        "video_url": f"http://localhost:5000/outputs/{output_filename}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
