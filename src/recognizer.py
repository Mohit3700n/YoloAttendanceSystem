import numpy as np
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self):
        # Sirf CPU par process karne ke liye CPUExecutionProvider explicitly diya hai
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # Det_size chota rakha hai taaki cropped face jaldi process ho
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        print("ArcFace Recognition Model Ready.")

    def get_embedding(self, face_crop):
        # InsightFace cropped image par analyze karega
        faces = self.app.get(face_crop)
        
        if len(faces) > 0:
            # Pehla face jo detect hua uska 512-D vector return karo
            return faces[0].embedding
        return None