from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load both models
ppe_model = YOLO("foodsafety.pt")
kitchen_model = YOLO("kitchenobj.pt")

image_path = "rename.jpg"   # change image here

# Run inference
ppe_results = ppe_model(image_path, conf=0.25)
kitchen_results = kitchen_model(image_path, conf=0.25)

# Get annotated images
ppe_annotated = ppe_results[0].plot()
kitchen_annotated = kitchen_results[0].plot()

# Combine detections on original image
combined_image = cv2.imread(image_path)

# Draw PPE boxes
for box in ppe_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls)
    label = ppe_results[0].names[cls_id]
    conf = float(box.conf)

    cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(combined_image, f"PPE: {label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2)

# Draw Kitchen boxes
for box in kitchen_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls)
    label = kitchen_results[0].names[cls_id]
    conf = float(box.conf)

    cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(combined_image, f"Hazard: {label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2)

# Convert BGR to RGB for matplotlib
combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(combined_image)
plt.axis("off")
plt.show()

print("Dual-model inference complete.")
# import os
# print(os.getcwd())