"""
Face Recognition Module - Task 5
CODSOFT AI Internship

Basic face recognition using feature comparison
"""

import cv2
import numpy as np
import os


class FaceRecognizer:
    """Simple face recognition using LBPH (Local Binary Patterns Histograms)"""
    
    def __init__(self):
        """Initialize face recognizer"""
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_faces = {}
        self.is_trained = False
    
    def add_face(self, image_path, person_name):
        """
        Add a face to the recognition database
        
        Args:
            image_path: Path to the image
            person_name: Name of the person
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Use first detected face
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        
        # Store face
        if person_name not in self.known_faces:
            self.known_faces[person_name] = []
        self.known_faces[person_name].append(face_roi)
    
    def train(self):
        """Train the recognizer with known faces"""
        if not self.known_faces:
            raise ValueError("No faces added for training")
        
        faces = []
        labels = []
        label_map = {}
        
        for idx, (name, face_list) in enumerate(self.known_faces.items()):
            label_map[idx] = name
            for face in face_list:
                faces.append(face)
                labels.append(idx)
        
        self.recognizer.train(faces, np.array(labels))
        self.label_map = label_map
        self.is_trained = True
    
    def recognize(self, image_path):
        """
        Recognize faces in an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            list: List of (name, confidence) tuples for each detected face
        """
        if not self.is_trained:
            raise ValueError("Recognizer not trained. Call train() first.")
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(image, 1.1, 5)
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            label, confidence = self.recognizer.predict(face_roi)
            name = self.label_map.get(label, "Unknown")
            results.append((name, confidence))
        
        return results
