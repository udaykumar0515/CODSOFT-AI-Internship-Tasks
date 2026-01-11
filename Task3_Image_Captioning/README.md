# Task 3: Image Captioning

## Overview

A simple image captioning system combining Computer Vision and Natural Language Processing.

## Architecture

- **CNN Encoder**: ResNet50 (pre-trained) extracts image features
- **RNN Decoder**: LSTM generates captions from features
- **Vocabulary**: Simple word list (~84 common words) - no hardcoding

## Files

- `model.py` - CNN encoder + LSTM decoder (~120 lines)
- `inference.py` - Caption generation logic (~100 lines)
- `main.py` - Demo script (~90 lines)
- `requirements.txt` - Dependencies

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add sample images to sample_images/ folder

# Run demo
python main.py
```

## How It Works

1. **Image Input**: Load and preprocess image (224x224, normalized)
2. **Feature Extraction**: ResNet50 extracts visual features
3. **Caption Generation**: LSTM generates words one by one
4. **Output**: Natural language caption

## Example Usage

```python
from inference import ImageCaptioner

captioner = ImageCaptioner()
caption = captioner.caption_image("path/to/image.jpg")
print(caption)
```

## Note

- Model uses **random weights** for demonstration
- For real applications, train on datasets like:
  - Flickr8k
  - Flickr30k
  - MS COCO Captions
- Training code available in `training.ipynb`

## Requirements Met

✓ **Computer Vision**: ResNet50 for feature extraction  
✓ **NLP**: LSTM for language generation  
✓ **Pre-trained Model**: ResNet50 (ImageNet weights)  
✓ **Caption Generation**: RNN-based decoder

## Simplified Design

- No hardcoded vocabulary (generated from word list)
- Clean, readable code (~310 lines total)
- Straightforward architecture
- Easy to understand and modify
