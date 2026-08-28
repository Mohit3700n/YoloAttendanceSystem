import cv2
import numpy as np
import os

class LivenessDetector:
    def __init__(self, model_path="../models/fas.onnx"):
        self.has_model = os.path.exists(model_path)
        self.input_size = (80, 80) # Default fallback
        
        if self.has_model:
            try:
                import onnxruntime as ort
                # CPU optimized execution
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                input_info = self.session.get_inputs()[0]
                self.input_name = input_info.name
                
                # PROFESSIONAL FIX: Dynamically fetch required input shape from the model
                shape = input_info.shape
                if len(shape) == 4:
                    # Handle NCHW format
                    if shape[1] == 3:
                        self.input_size = (shape[3], shape[2])
                    else: # Handle NHWC format
                        self.input_size = (shape[2], shape[1])
                
                print(f"Liveness Model Loaded. Automatically set input size to: {self.input_size}")
                
            except ImportError:
                print("Error: onnxruntime installed nahi hai.")
                exit()
        else:
            print(f"Warning: Liveness model '{model_path}' nahi mila.")
            print("=> Liveness check BYPASS mode me hai.")

    def check(self, face_crop):
        if not self.has_model:
            return True, 0.99  # Bypass
            
        try:
            # Ab image dynamically resize hogi us size par jo model ko chahiye
            face_resized = cv2.resize(face_crop, self.input_size)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            img_blob = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
            img_blob = np.expand_dims(img_blob, axis=0) / 255.0
            
            outputs = self.session.run(None, {self.input_name: img_blob})
            preds = outputs[0][0] # Yeh raw logits hain (jaise -1.10 ya -8.48)
            
            # PROFESSIONAL FIX: Apply Softmax function
            # Yeh raw negative/positive numbers ko 0.0 se 1.0 ki probability me convert karega
            exp_preds = np.exp(preds - np.max(preds)) # max subtract kiya numerical stability ke liye
            probs = exp_preds / exp_preds.sum()
            
            # Index 0 Real Face hai (Kyunki real face par -1.10 bada score tha)
            real_prob = float(probs[0])
            
            # Threshold 0.75 rakho
            is_real = real_prob > 0.60
            return is_real, real_prob
            
        except Exception as e:
            print(f"\n[LIVENESS ERROR]: {e}\n")
            return False, 0.0