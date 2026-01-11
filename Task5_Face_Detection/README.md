# Task 5: Face Detection and Recognition

## Overview

AI application for detecting and recognizing faces in images using Haar Cascade classifier and LBPH face recognition.

## 🎥 Demo Video

Watch face detection in action: [View Demo](../demo_videos/task5.mp4)

## Features

- **Face Detection**: Haar Cascade classifier (pre-trained)
- **Face Recognition**: LBPH (Local Binary Patterns Histograms)
- **Simple Interface**: Process images and save results
- **OpenCV-based**: Uses industry-standard computer vision library

## Files

- `main.py` - Main application
- `face_detector.py` - Face detection using Haar Cascade
- `face_recognizer.py` - Face recognition using LBPH
- `requirements.txt` - Dependencies

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add images to sample_images/ folder

# Run detection
python main.py
```

## Usage

### Face Detection

```python
from face_detector import FaceDetector

detector = FaceDetector()
image, faces = detector.detect_faces("image.jpg")
detector.save_result(image, "output.jpg")
```

### Face Recognition (Optional)

```python
from face_recognizer import FaceRecognizer

recognizer = FaceRecognizer()
recognizer.add_face("person1.jpg", "John")
recognizer.train()
results = recognizer.recognize("test.jpg")
```

## How It Works

1. **Load Image**: Read image from file
2. **Detect Faces**: Haar Cascade finds faces
3. **Draw Rectangles**: Mark detected faces
4. **Save Result**: Output image with markings

## Technical Details

- **Detection**: Haar Cascade (pre-trained on faces)
- **Recognition**: LBPH algorithm
- **Library**: OpenCV (cv2)
- **Input**: Images (JPG, PNG, BMP)
- **Output**: Images with detected faces marked

## Requirements Met

✓ **Face Detection**: Haar Cascade classifier  
✓ **Pre-trained Model**: Uses OpenCV's trained cascades  
✓ **Face Recognition**: LBPH algorithm  
✓ **Image Processing**: Detect faces and mark them

## Output

- Detected faces marked with green rectangles
- Results saved in `detected_faces/` directory
- Console output shows number of faces found
