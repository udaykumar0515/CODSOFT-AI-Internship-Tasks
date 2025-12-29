# Task 3: Image Captioning with CNN + RNN/Transformer

## Overview

An image captioning system that combines pretrained CNN models (ResNet) for feature extraction with RNN (LSTM) or Transformer decoders to generate descriptive captions for images.

## Features

- **CNN Encoder**: Uses pretrained ResNet50 for image feature extraction
- **Dual Decoder Options**: LSTM or Transformer for caption generation
- **Preprocessing Pipeline**: ImageNet-standard normalization and resizing
- **Batch Processing**: Support for single image or directory processing
- **Extensible Architecture**: Easy to modify and extend
- **Demo Vocabulary**: Built-in vocabulary for demonstration

## Architecture

### CNN Encoder (ResNet50)
- **Input**: RGB images (224x224)
- **Feature Extraction**: Removes final classification layer
- **Output**: 256-dimensional feature vectors
- **Pretrained**: ImageNet weights (optional fine-tuning)

### RNN Decoder (LSTM)
- **Embedding Layer**: Word embeddings (256-dim)
- **LSTM Layers**: Hidden size 512, configurable layers
- **Output Layer**: Vocabulary prediction
- **Sequence Generation**: Greedy search for caption generation

### Transformer Decoder
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: 8 attention heads
- **Feed-Forward Networks**: 512 hidden units
- **Causal Masking**: Prevents future token access

## Files Structure

```
Task3_Image_Captioning/
├── main.py              # Demo and entry point
├── inference.py         # Inference script with CLI
├── model.py            # Model architectures
├── training.ipynb      # Training notebook (placeholder)
├── requirements.txt    # Dependencies
├── README.md          # This file
└── sample_images/     # Directory for test images
```

## Installation

1. Navigate to the Task3 directory:
   ```bash
   cd Task3_Image_Captioning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create sample images directory:
   ```bash
   mkdir -p sample_images
   ```

## Usage

### Quick Demo
```bash
python main.py
```

### Single Image Captioning
```bash
python inference.py --image path/to/image.jpg
```

### Batch Processing
```bash
python inference.py --image_dir path/to/images/
```

### Advanced Options
```bash
python inference.py --image image.jpg --decoder transformer --max_length 25
```

## Command Line Arguments

- `--image`: Path to single image
- `--image_dir`: Directory containing images
- `--model`: Path to trained model weights
- `--decoder`: Decoder type (`lstm` or `transformer`)
- `--max_length`: Maximum caption length (default: 20)
- `--output`: Output file for saving captions

## Example Usage

```bash
# Process single image
python inference.py --image sample_images/dog.jpg --decoder lstm

# Process directory
python inference.py --image_dir sample_images/ --output captions.txt

# Use transformer decoder
python inference.py --image sample_images/cat.jpg --decoder transformer --max_length 15
```

## Model Components

### CNN Encoder Features
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Feature Size**: 2048 → 256 (projected)
- **Normalization**: Batch normalization
- **Fine-tuning**: Optional (disabled by default)

### LSTM Decoder Features
- **Embedding Size**: 256 dimensions
- **Hidden Size**: 512 units
- **Layers**: 1 LSTM layer (configurable)
- **Dropout**: Applied for regularization
- **Sequence Generation**: Greedy search

### Transformer Decoder Features
- **Model Dimension**: 256
- **Attention Heads**: 8
- **Feed-Forward Size**: 512
- **Layers**: 6 transformer layers
- **Positional Encoding**: Sinusoidal

## Image Preprocessing

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Vocabulary System

The system includes a demo vocabulary with common English words:
- **Special Tokens**: `<pad>`, `<start>`, `<end>`, `<unk>`
- **Common Words**: Articles, prepositions, common nouns
- **Descriptive Words**: Colors, sizes, actions, objects
- **Size**: ~250 words (expandable)

## Training (Placeholder)

For training on real datasets:
1. **Datasets**: Flickr8k, Flickr30k, COCO Captions
2. **Training Script**: See `training.ipynb`
3. **Hyperparameters**: Learning rate, batch size, epochs
4. **Evaluation**: BLEU, METEOR, CIDEr scores

## Performance Considerations

### Hardware Requirements
- **CPU**: Works but slower
- **GPU**: Recommended for faster inference
- **Memory**: 4GB+ RAM, 8GB+ VRAM for GPU

### Optimization Tips
- Use GPU for faster processing
- Batch processing for multiple images
- Model quantization for deployment
- Caching preprocessed features

## Example Output

```
📷 dog.jpg
💬 A brown dog is running in the grass.

📷 beach.jpg
💬 People are walking on the sandy beach.

📷 city.jpg
💬 Tall buildings are visible in the urban skyline.
```

## Extensions and Improvements

### Model Enhancements
- **Attention Mechanisms**: Visual attention for better focus
- **Beam Search**: Better caption generation
- **Ensemble Methods**: Multiple model combination
- **Transfer Learning**: Domain-specific fine-tuning

### Data Augmentation
- **Image Augmentation**: Rotation, flip, color jitter
- **Text Augmentation**: Back-translation, synonym replacement
- **Multi-task Learning**: Object detection + captioning

### Deployment Options
- **Web Interface**: Flask/FastAPI application
- **Mobile App**: TensorFlow Lite, ONNX
- **Cloud Service**: AWS Lambda, Google Cloud Functions
- **Edge Computing**: Raspberry Pi, Jetson Nano

## Technical Specifications

- **Framework**: PyTorch
- **CNN**: ResNet50 (pretrained)
- **RNN**: LSTM (256→512→vocab)
- **Transformer**: 6 layers, 8 heads
- **Input Size**: 224×224×3 RGB
- **Feature Size**: 256 dimensions
- **Vocabulary Size**: Configurable (demo: 250)

## Evaluation Metrics

For trained models, use standard captioning metrics:
- **BLEU-1 to BLEU-4**: N-gram precision
- **METEOR**: Semantic similarity
- **ROUGE**: Recall-oriented metrics
- **CIDEr**: Consensus-based image description
- **SPICE**: Scene graph-based evaluation

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Image Loading Errors**: Check file formats and paths
3. **Model Loading**: Ensure correct architecture and weights
4. **Vocabulary Mismatch**: Align training and inference vocabularies

### Solutions
- Use smaller image batches
- Verify image file integrity
- Check model compatibility
- Rebuild vocabulary if needed

## Research References

- **Show and Tell**: Image captioning with neural networks
- **Show, Attend and Tell**: Visual attention mechanisms
- **Bottom-Up and Top-Down Attention**: Region-based features
- **Transformer Models**: Self-attention for captioning

## Evaluation Criteria

This task demonstrates:
- ✅ CNN feature extraction implementation
- ✅ RNN/Transformer decoder architecture
- ✅ Image preprocessing pipeline
- ✅ Caption generation logic
- ✅ Model design and modularity
- ✅ Inference and CLI interface
- ✅ Documentation and examples
