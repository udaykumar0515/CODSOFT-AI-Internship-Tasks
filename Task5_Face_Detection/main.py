"""
Face Detection and Recognition - Task 5
CODSOFT AI Internship

Main application for face detection and recognition
"""

import os
from face_detector import FaceDetector


def main():
    """Main function for face detection demo"""
    print("=" * 60)
    print("FACE DETECTION AND RECOGNITION - TASK 5")
    print("=" * 60)
    print("\nFace Detection using Haar Cascade Classifier")
    print("=" * 60)
    
    # Initialize detector
    print("\nInitializing face detector...")
    try:
        detector = FaceDetector()
        print("Face detector initialized successfully!")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Check for sample images
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\nPlease add images to '{sample_dir}' directory")
        return
    
    # Get images
    image_files = [f for f in os.listdir(sample_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"\nNo images found in '{sample_dir}'")
        return
    
    print(f"\nFound {len(image_files)} image(s)")
    
    # Process each image
    output_dir = "detected_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in image_files:
        img_path = os.path.join(sample_dir, img_file)
        print(f"\nProcessing: {img_file}")
        
        try:
            # Detect faces
            result_img, faces = detector.detect_faces(img_path)
            
            print(f"  Detected {len(faces)} face(s)")
            
            # Save result
            output_path = os.path.join(output_dir, f"detected_{img_file}")
            detector.save_result(result_img, output_path)
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Face detection complete!")
    print(f"Results saved in '{output_dir}' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
