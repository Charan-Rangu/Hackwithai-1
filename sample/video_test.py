from ultralytics import YOLO
import cv2
import os

ppe_model = YOLO("foodsafety.pt")
kitchen_model = YOLO("kitchenobj.pt")

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(OUTPUT_FOLDER, "processed_video.mp4")

    # 🔥 USE H264 COMPATIBLE CODEC
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ppe_results = ppe_model(frame, conf=0.25)
        kitchen_results = kitchen_model(frame, conf=0.25)

        # Draw PPE (Green)
        for box in ppe_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = ppe_results[0].names[int(box.cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"PPE: {label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw Hazards (Red)
        for box in kitchen_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = kitchen_results[0].names[int(box.cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f"Hazard: {label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        out.write(frame)

    cap.release()
    out.release()

    print("Video processing complete.")

    return output_path
    