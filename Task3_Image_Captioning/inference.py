"""
Image Captioning Inference Script
CODSOFT AI Internship - Task 3
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import ImageCaptioningModel
import argparse
import os
from typing import List, Dict


class ImageCaptioningInference:
    """Inference class for image captioning"""
    
    def __init__(self, model_path: str = None, decoder_type: str = 'lstm',
                 vocab_size: int = 1000, embed_size: int = 256, hidden_size: int = 512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_type = decoder_type
        
        # Initialize model
        self.model = ImageCaptioningModel(
            encoder_type='resnet',
            decoder_type=decoder_type,
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            fine_tune_encoder=False
        ).to(self.device)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No pretrained weights loaded. Using random initialization.")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Simple vocabulary (for demonstration)
        self.vocab = self._create_simple_vocab()
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
    
    def _create_simple_vocab(self) -> Dict[str, int]:
        """Create a simple vocabulary for demonstration"""
        vocab = {
            '<pad>': 0,
            '<start>': 1,
            '<end>': 2,
            '<unk>': 3,
            'a': 4,
            'the': 5,
            'is': 6,
            'are': 7,
            'dog': 8,
            'cat': 9,
            'person': 10,
            'man': 11,
            'woman': 12,
            'child': 13,
            'car': 14,
            'tree': 15,
            'house': 16,
            'building': 17,
            'street': 18,
            'road': 19,
            'sky': 20,
            'cloud': 21,
            'sun': 22,
            'moon': 23,
            'water': 24,
            'ocean': 25,
            'beach': 26,
            'mountain': 27,
            'forest': 28,
            'flower': 29,
            'grass': 30,
            'park': 31,
            'city': 32,
            'town': 33,
            'bridge': 34,
            'boat': 35,
            'plane': 36,
            'bird': 37,
            'horse': 38,
            'cow': 39,
            'sheep': 40,
            'pig': 41,
            'chicken': 42,
            'dog': 43,
            'cat': 44,
            'red': 45,
            'blue': 46,
            'green': 47,
            'yellow': 48,
            'black': 49,
            'white': 50,
            'brown': 51,
            'gray': 52,
            'orange': 53,
            'purple': 54,
            'pink': 55,
            'big': 56,
            'small': 57,
            'large': 58,
            'little': 59,
            'tall': 60,
            'short': 61,
            'long': 62,
            'wide': 63,
            'narrow': 64,
            'round': 65,
            'square': 66,
            'running': 67,
            'walking': 68,
            'sitting': 69,
            'standing': 70,
            'jumping': 71,
            'playing': 72,
            'eating': 73,
            'sleeping': 74,
            'working': 75,
            'driving': 76,
            'flying': 77,
            'swimming': 78,
            'climbing': 79,
            'in': 80,
            'on': 81,
            'at': 82,
            'by': 83,
            'with': 84,
            'and': 85,
            'or': 86,
            'but': 87,
            'has': 88,
            'have': 89,
            'had': 90,
            'was': 91,
            'were': 92,
            'been': 93,
            'being': 94,
            'am': 95,
            'is': 96,
            'are': 97,
            'was': 98,
            'were': 99,
            'be': 100,
            'have': 101,
            'has': 102,
            'had': 103,
            'do': 104,
            'does': 105,
            'did': 106,
            'will': 107,
            'would': 108,
            'could': 109,
            'should': 110,
            'may': 111,
            'might': 112,
            'must': 113,
            'can': 114,
            'this': 115,
            'that': 116,
            'these': 117,
            'those': 118,
            'i': 119,
            'you': 120,
            'he': 121,
            'she': 122,
            'it': 123,
            'we': 124,
            'they': 125,
            'me': 126,
            'him': 127,
            'her': 128,
            'us': 129,
            'them': 130,
            'my': 131,
            'your': 132,
            'his': 133,
            'her': 134,
            'its': 135,
            'our': 136,
            'their': 137,
            'mine': 138,
            'yours': 139,
            'hers': 140,
            'ours': 141,
            'theirs': 142,
            'of': 143,
            'for': 144,
            'from': 145,
            'to': 146,
            'into': 147,
            'onto': 148,
            'upon': 149,
            'over': 150,
            'under': 151,
            'below': 152,
            'above': 153,
            'between': 154,
            'among': 155,
            'through': 156,
            'across': 157,
            'around': 158,
            'near': 159,
            'far': 160,
            'here': 161,
            'there': 162,
            'everywhere': 163,
            'nowhere': 164,
            'somewhere': 165,
            'anywhere': 166,
            'up': 167,
            'down': 168,
            'left': 169,
            'right': 170,
            'forward': 171,
            'backward': 172,
            'inside': 173,
            'outside': 174,
            'indoor': 175,
            'outdoor': 176,
            'indoor': 177,
            'outdoor': 178,
            'natural': 179,
            'artificial': 180,
            'modern': 181,
            'old': 182,
            'new': 183,
            'young': 184,
            'ancient': 185,
            'contemporary': 186,
            'traditional': 187,
            'urban': 188,
            'rural': 189,
            'suburban': 190,
            'industrial': 191,
            'commercial': 192,
            'residential': 193,
            'agricultural': 194,
            'recreational': 195,
            'educational': 196,
            'medical': 197,
            'government': 198,
            'military': 199,
            'religious': 200,
            'cultural': 201,
            'historical': 202,
            'tourist': 203,
            'local': 204,
            'international': 205,
            'national': 206,
            'regional': 207,
            'global': 208,
            'public': 209,
            'private': 210,
            'personal': 211,
            'professional': 212,
            'business': 213,
            'leisure': 214,
            'sport': 215,
            'art': 216,
            'music': 217,
            'dance': 218,
            'theater': 219,
            'cinema': 220,
            'television': 221,
            'radio': 222,
            'internet': 223,
            'computer': 224,
            'phone': 225,
            'camera': 226,
            'book': 227,
            'magazine': 228,
            'newspaper': 229,
            'food': 230,
            'drink': 231,
            'fruit': 232,
            'vegetable': 233,
            'meat': 234,
            'fish': 235,
            'bread': 236,
            'rice': 237,
            'pasta': 238,
            'salad': 239,
            'soup': 240,
            'dessert': 241,
            'coffee': 242,
            'tea': 243,
            'juice': 244,
            'water': 245,
            'milk': 246,
            'beer': 247,
            'wine': 248,
            'whiskey': 249,
            'vodka': 250,
        }
        return vocab
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def generate_caption(self, image_path: str, max_length: int = 20) -> str:
        """Generate caption for an image"""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return "Error: Could not process image"
        
        # Generate caption
        with torch.no_grad():
            caption_ids = self.model.generate_caption(image_tensor, max_len=max_length)
        
        # Convert IDs to words
        caption_words = []
        for idx in caption_ids:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word == '<end>':
                    break
                elif word not in ['<start>', '<pad>', '<unk>']:
                    caption_words.append(word)
        
        # Format caption
        if caption_words:
            caption = ' '.join(caption_words)
            caption = caption.capitalize() + '.'
            return caption
        else:
            return "Unable to generate caption."
    
    def generate_batch_captions(self, image_paths: List[str], 
                              max_length: int = 20) -> List[str]:
        """Generate captions for multiple images"""
        captions = []
        for image_path in image_paths:
            caption = self.generate_caption(image_path, max_length)
            captions.append(caption)
            print(f"Processed: {image_path} -> {caption}")
        return captions


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Image Captioning Inference')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--decoder', type=str, default='lstm', 
                       choices=['lstm', 'transformer'], help='Decoder type')
    parser.add_argument('--max_length', type=int, default=20, 
                       help='Maximum caption length')
    parser.add_argument('--output', type=str, help='Output file for captions')
    
    args = parser.parse_args()
    
    # Initialize inference
    print("Initializing Image Captioning Model...")
    inference = ImageCaptioningInference(
        model_path=args.model,
        decoder_type=args.decoder
    )
    
    # Process images
    if args.image:
        # Single image
        print(f"\nGenerating caption for: {args.image}")
        caption = inference.generate_caption(args.image, args.max_length)
        print(f"Caption: {caption}")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"{args.image}: {caption}\n")
            print(f"Caption saved to: {args.output}")
    
    elif args.image_dir:
        # Directory of images
        print(f"\nProcessing images in directory: {args.image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for filename in os.listdir(args.image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.image_dir, filename))
        
        if not image_paths:
            print("No image files found in the directory.")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Generate captions
        captions = inference.generate_batch_captions(image_paths, args.max_length)
        
        # Display results
        print("\n" + "="*60)
        print("GENERATED CAPTIONS")
        print("="*60)
        for img_path, caption in zip(image_paths, captions):
            print(f"{os.path.basename(img_path)}: {caption}")
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                for img_path, caption in zip(image_paths, captions):
                    f.write(f"{os.path.basename(img_path)}: {caption}\n")
            print(f"\nCaptions saved to: {args.output}")
    
    else:
        print("Please specify either --image or --image_dir")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
