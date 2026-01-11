"""
Face Detection Module - Task 5
CODSOFT AI Internship

Face detection using Haar Cascade classifier
"""

import cv2
import os


class FaceDetector:
    """Detect faces in images using Haar Cascade"""
    
    def __init__(self):
        """Initialize face detector with Haar Cascade"""
        # Load pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (image with detected faces, list of face coordinates)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return image, faces
    
    def save_result(self, image, output_path):
        """Save the image with detected faces"""
        cv2.imwrite(output_path, image)
