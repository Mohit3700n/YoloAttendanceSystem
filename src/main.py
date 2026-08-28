import cv2
import time
import numpy as np
import threading
from ultralytics import YOLO
from db_manager import DBManager
from recognizer import FaceRecognizer
from liveness import LivenessDetector

# --- Helper Function ---
def compute_cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    return dot_product / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

print("Initializing System Models...")
recognizer = FaceRecognizer()
liveness_model = LivenessDetector()
try:
    detector = YOLO("../models/yolov8n-face.pt")
except:
    print("YOLO model fail.")
    exit()

# Memory load
temp_db = DBManager()
registered_users = temp_db.load_all_users()
temp_db.conn.close()
print(f"Loaded {len(registered_users)} users from database.")

MATCH_THRESHOLD = 0.45
COOLDOWN_SECONDS = 60
last_marked = {} 

# --- SHARED VARIABLES FOR THREADING ---
shared_frame = None
shared_results = []
is_running = True

# --- BACKGROUND AI WORKER THREAD ---
def ai_processing_worker():
    global shared_frame, shared_results, is_running
    
    # Thread ke andar alag DB connection chahiye hota hai SQLite me
    db = DBManager() 
    
    while is_running:
        if shared_frame is None:
            time.sleep(0.01)
            continue
            
        # Current frame ki copy lo taaki main thread usko overwrite na kare
        frame_to_process = shared_frame.copy()
        current_results = []
        
        results = detector(frame_to_process, stream=True, verbose=False, imgsz=320)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                h, w, _ = frame_to_process.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    # 1. Liveness ke liye box ko 40% bada karo taaki background context mile
                    box_w = x2 - x1
                    box_h = y2 - y1
                    pad_w = int(box_w * 0.4)
                    pad_h = int(box_h * 0.4)
                    
                    lx1, ly1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                    lx2, ly2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                    
                    liveness_crop = frame_to_process[ly1:ly2, lx1:lx2]
                    
                    if liveness_crop.size != 0:
                        is_real, conf = liveness_model.check(liveness_crop)
                        
                        if not is_real:
                            # Debugging ke liye spoof score bhi UI par dikhao
                            current_results.append(((x1, y1, x2, y2), (0, 0, 255), f"SPOOF ({conf:.2f})"))
                            continue
                            
                        # 2. Recognition ke liye original tight YOLO crop hi use karo
                        face_crop = frame_to_process[y1:y2, x1:x2]
                        live_embedding = recognizer.get_embedding(face_crop)
                        
                        if live_embedding is not None:
                            best_match_name = "Unknown"
                            best_score = -1
                            
                            for user in registered_users:
                                score = compute_cosine_similarity(live_embedding, user['embedding'])
                                if score > best_score:
                                    best_score = score
                                    best_match_name = user['name']
                            
                            if best_score >= MATCH_THRESHOLD:
                                current_time = time.time()
                                if best_match_name not in last_marked or (current_time - last_marked[best_match_name]) > COOLDOWN_SECONDS:
                                    db.cursor.execute("INSERT INTO attendance_logs (user_id) VALUES ((SELECT id FROM users WHERE name=?))", (best_match_name,))
                                    db.conn.commit()
                                    last_marked[best_match_name] = current_time
                                    print(f"✅ Attendance Marked: {best_match_name} (Score: {best_score:.2f})")
                                    
                                current_results.append(((x1, y1, x2, y2), (0, 255, 0), f"{best_match_name} ({best_score:.2f})"))
                            else:
                                current_results.append(((x1, y1, x2, y2), (0, 165, 255), f"Unknown ({best_score:.2f})"))
        
        # UI loop ke liye results update karo
        shared_results = current_results
        
        # CPU ko thoda saans lene do taaki fan blast na ho
        time.sleep(0.05) 
        
    db.conn.close()

ai_thread = threading.Thread(target=ai_processing_worker)
ai_thread.daemon = True  # FIX 1: Daemon thread - Main process close hote hi yeh turant kill ho jayega
ai_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("--- SYSTEM LIVE --- (Press 'q' or Ctrl+C to exit)")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Share frame to background thread
        shared_frame = frame.copy()
        
        # Draw latest available results from background thread
        for (box, color, label) in shared_results:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        cv2.imshow("Live Attendance System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

except KeyboardInterrupt:
    # FIX 2: Catch Ctrl+C cleanly
    print("\n[INFO] KeyboardInterrupt received. Shutting down...")
    is_running = False

finally:
    # Yeh block hamesha execute hoga, chahe 'q' dabao ya 'Ctrl+C'
    print("[INFO] Cleaning up camera and memory resources...")
    cap.release()
    cv2.destroyAllWindows()
    # FIX 3: Join par 1 second ka timeout lagaya hai taaki infinite stuck na ho
    ai_thread.join(timeout=1.0) 
    print("[INFO] System shutdown complete.")