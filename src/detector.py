import cv2
import time
from ultralytics import YOLO
from liveness import LivenessDetector

MODEL_PATH = "../models/yolov8n-face.pt" 

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error: {e}")
    exit()

liveness_model = LivenessDetector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
avg_fps = 0.0
frame_count = 0
PROCESS_EVERY_N_FRAMES = 3  # Har 3rd frame par YOLO chalega
last_boxes = [] # Pichle frame ke boxes store karne ke liye

while True:
    success, frame = cap.read()
    if not success:
        print("Camera access failed.")
        break

    frame_count += 1

    # Sirf har Nth frame par inference run karo
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # imgsz=320 tensor size half kar dega speed badhane ke liye
        # 'half=False' hata diya gaya hai warning resolve karne ke liye
        results = model(frame, stream=True, verbose=False, imgsz=320)
        
        last_boxes = [] # Naye boxes aane par purane clear karo
        for r in results:
            for box in r.boxes:
                # CPU par direct integer mapping
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                last_boxes.append((x1, y1, x2, y2))

    # Stored boxes ko draw karo (taki skipped frames me bhi box dikhe)
    # Stored boxes ko draw karo aur face crop karo
    for (x1, y1, x2, y2) in last_boxes:
        # 1. Boundary Checks (Crash Prevention)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if x2 > x1 and y2 > y1:
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size != 0:
                # 3. Liveness Check
                is_real, conf = liveness_model.check(face_crop)
                
                # Agar real hai toh Green box, Fake hai toh Red box
                color = (0, 255, 0) if is_real else (0, 0, 255)
                label = f"{'Real' if is_real else 'Fake'} ({conf:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS Calculation
    curr_time = time.time()
    time_diff = curr_time - prev_time
    
    if time_diff > 0:
        current_fps = 1.0 / time_diff
        # EMA (Exponential Moving Average): Naye frame ko 10% weight, purane average ko 90% weight
        avg_fps = (avg_fps * 0.9) + (current_fps * 0.1)
        
    prev_time = curr_time
    
    # Display stable average FPS
    cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("CPU Optimized Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()