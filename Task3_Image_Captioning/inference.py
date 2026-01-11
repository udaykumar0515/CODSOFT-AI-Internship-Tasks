"""
Simple Image Captioning Inference
Task 3 - CODSOFT AI Internship

Simple inference script for generating captions from images
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ImageCaptioningModel


class ImageCaptioner:
    """Simple class to caption images"""
    
    def __init__(self):
        """Initialize the model and vocabulary"""
        print("Initializing captioning model...")
        
        # Create a simple vocabulary (no hardcoding, just common words)
        self.vocab = self._create_vocab()
        self.vocab_size = len(self.vocab)
        
        # Create model
        self.model = ImageCaptioningModel(
            embed_size=256,
            hidden_size=512,
            vocab_size=self.vocab_size
        )
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Model ready! Vocabulary size: {self.vocab_size}")
    
    def _create_vocab(self):
        """Create a simple vocabulary from common words"""
        # Basic vocabulary with common caption words
        words = [
            '<pad>', '<start>', '<end>', '<unk>',
            'a', 'an', 'the', 'is', 'are', 'and', 'in', 'on', 'with',
            'person', 'people', 'man', 'woman', 'child', 'dog', 'cat',
            'car', 'bike', 'bus', 'train', 'plane',
            'building', 'tree', 'grass', 'sky', 'water', 'beach', 'mountain',
            'standing', 'sitting', 'walking', 'running', 'playing',
            'street', 'road', 'park', 'field', 'city',
            'wearing', 'holding', 'next', 'front', 'behind',
            'red', 'blue', 'green', 'black', 'white', 'brown',
            'large', 'small', 'old', 'young', 'beautiful',
            'eating', 'drinking', 'looking', 'smiling',
            'group', 'riding', 'driving', 'jumping',
            'sunny', 'cloudy', 'day', 'night',
            'table', 'chair', 'food', 'ball', 'flower',
            'house', 'room', 'window', 'door',
            'horse', 'bird', 'animal', 'sign', 'snow'
        ]
        
        # Create word to index mapping
        vocab = {word: idx for idx, word in enumerate(words)}
        return vocab
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        return image
    
    def caption_image(self, image_path, max_length=15, temperature=1.2):
        """Generate a caption for an image"""
        try:
            # Load and preprocess image
            image = self.preprocess_image(image_path)
            
            # Generate caption with temperature for diversity
            with torch.no_grad():
                caption = self.model.generate_caption(image, self.vocab, max_length, temperature)
            
            # If caption is empty, return a default
            if not caption.strip():
                caption = "a scene"
            
            return caption
            
        except Exception as e:
            print(f"Error captioning {image_path}: {e}")
            return "image could not be processed"
    
    def caption_multiple(self, image_paths, max_length=15):
        """Generate captions for multiple images"""
        captions = []
        for img_path in image_paths:
            caption = self.caption_image(img_path, max_length)
            captions.append(caption)
        return captions
