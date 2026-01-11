"""
Simple Image Captioning Model
Task 3 - CODSOFT AI Internship

A straightforward implementation combining:
- CNN (ResNet50) for image feature extraction
- LSTM for caption generation
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """CNN encoder to extract image features using ResNet50"""
    
    def __init__(self, embed_size=256):
        super(ImageEncoder, self).__init__()
        # Use pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Linear layer to project features to embedding space
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        """Extract features from images"""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


class CaptionDecoder(nn.Module):
    """LSTM decoder to generate captions from image features"""
    
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=100):
        super(CaptionDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """Generate caption probabilities"""
        # Embed the captions
        embeddings = self.embed(captions)
        # Concatenate image features with caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # Pass through LSTM
        hiddens, _ = self.lstm(embeddings)
        # Get predictions
        outputs = self.linear(hiddens)
        return outputs


class ImageCaptioningModel(nn.Module):
    """Complete image captioning model"""
    
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=100):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size)
        
    def forward(self, images, captions):
        """Forward pass"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, image, vocab, max_length=20, temperature=1.0):
        """Generate caption for a single image with sampling diversity"""
        # Create reverse vocab (index to word)
        idx_to_word = {idx: word for word, idx in vocab.items()}
        
        # Get image features
        features = self.encoder(image)
        
        # Use image features hash for consistent but different sampling per image
        feature_hash = int(torch.sum(features).item() * 1000) % 1000
        torch.manual_seed(feature_hash)
        
        # Initialize with <start> token
        word_idx = vocab['<start>']
        caption = []
        
        # Initialize LSTM hidden state
        hidden = None
        
        # Generate words one by one
        for _ in range(max_length):
            # Get current word embedding
            word_tensor = torch.tensor([[word_idx]])
            embedding = self.decoder.embed(word_tensor)
            
            # For first iteration, use image features
            if hidden is None:
                lstm_input = torch.cat((features.unsqueeze(1), embedding), 1)
                hiddens, hidden = self.decoder.lstm(lstm_input)
            else:
                hiddens, hidden = self.decoder.lstm(embedding, hidden)
            
            # Get next word prediction
            output = self.decoder.linear(hiddens[:, -1, :])
            
            # Apply temperature and sample from distribution
            # Higher temperature = more diversity
            output = output / temperature
            probabilities = torch.softmax(output, dim=-1)
            
            # Sample from the distribution instead of always picking max
            word_idx = torch.multinomial(probabilities, 1).item()
            
            # Get word from vocab
            word = idx_to_word.get(word_idx, '<unk>')
            
            # Stop if <end> token or unknown word
            if word in ['<end>', '<unk>', '<pad>']:
                break
                
            caption.append(word)
            
        return ' '.join(caption) if caption else 'a scene'
