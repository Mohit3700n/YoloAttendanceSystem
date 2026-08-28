import cv2
from db_manager import DBManager
from recognizer import FaceRecognizer
from ultralytics import YOLO

def enroll_new_person(person_name):
    print("Loading models...")
    db = DBManager()
    recognizer = FaceRecognizer()
    detector = YOLO("../models/yolov8n-face.pt")
    
    cap = cv2.VideoCapture(0)
    print(f"Camera started. Looking for {person_name}...")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        results = detector(frame, stream=True, verbose=False, imgsz=320)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                x1, y1 = max(0, x1), max(0, y1)
                
                # Face Box Draw Karo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Press 'S' to Save Face", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size != 0:
                        print("Generating embedding...")
                        embedding = recognizer.get_embedding(face_crop)
                        
                        if embedding is not None:
                            db.register_user(person_name, embedding)
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        else:
                            print("ArcFace face detect nahi kar paya. Dobara try karo.")

        cv2.imshow("Enrollment Phase", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter the name of the person to enroll: ")
    enroll_new_person(name)