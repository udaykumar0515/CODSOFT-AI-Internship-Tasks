"""
Image Captioning Model - CNN Encoder + RNN/Transformer Decoder
CODSOFT AI Internship - Task 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Tuple, Optional
import math


class CNNEncoder(nn.Module):
    """CNN Encoder using pretrained ResNet for feature extraction"""
    
    def __init__(self, embed_size: int = 256, fine_tune: bool = False):
        super(CNNEncoder, self).__init__()
        self.embed_size = embed_size
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Add linear layer to project to embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        # Freeze ResNet parameters if not fine-tuning
        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN encoder"""
        with torch.no_grad():
            features = self.resnet(images)  # (batch_size, 2048, 7, 7)
        
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        features = self.linear(features)  # (batch_size, embed_size)
        features = self.bn(features)  # (batch_size, embed_size)
        
        return features


class RNNDecoder(nn.Module):
    """RNN Decoder using LSTM for caption generation"""
    
    def __init__(self, embed_size: int = 256, hidden_size: int = 512, 
                 vocab_size: int = 1000, num_layers: int = 1):
        super(RNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer for vocabulary
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM decoder
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features: torch.Tensor, captions: torch.Tensor, 
                lengths: list) -> torch.Tensor:
        """Forward pass through RNN decoder"""
        # Embed captions
        embeddings = self.embed(captions)  # (batch_size, seq_length, embed_size)
        
        # Concatenate CNN features and caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        hiddens, _ = self.lstm(packed)
        
        # Unpack sequence
        outputs = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)[0]
        
        # Generate vocabulary scores
        outputs = self.linear(outputs)  # (batch_size, seq_length, vocab_size)
        
        return outputs
    
    def sample(self, features: torch.Tensor, states: Optional[Tuple] = None, 
               max_len: int = 20) -> list:
        """Generate caption using greedy search"""
        sampled_ids = []
        
        # Initialize input with CNN features
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        
        for i in range(max_len):
            # LSTM forward pass
            hiddens, states = self.lstm(inputs, states)  # (1, 1, hidden_size)
            
            # Generate output
            outputs = self.linear(hiddens.squeeze(1))  # (1, vocab_size)
            
            # Get most likely word
            _, predicted = outputs.max(1)  # (1,)
            sampled_ids.append(predicted.item())
            
            # Stop if <end> token is generated
            if predicted.item() == 2:  # Assuming 2 is <end> token
                break
            
            # Use predicted word as next input
            inputs = self.embed(predicted)  # (1, embed_size)
            inputs = inputs.unsqueeze(1)  # (1, 1, embed_size)
        
        return sampled_ids


class TransformerDecoder(nn.Module):
    """Transformer Decoder for caption generation"""
    
    def __init__(self, embed_size: int = 256, hidden_size: int = 512, 
                 vocab_size: int = 1000, num_heads: int = 8, 
                 num_layers: int = 6, max_seq_length: int = 50):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_seq_length)
        
        # Feature projection
        self.feature_projection = nn.Linear(embed_size, embed_size)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # Output layer
        self.output_projection = nn.Linear(embed_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, features: torch.Tensor, captions: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Transformer decoder"""
        batch_size = features.size(0)
        
        # Project CNN features
        memory = self.feature_projection(features.unsqueeze(1))  # (batch_size, 1, embed_size)
        
        # Embed captions and add positional encoding
        caption_embeddings = self.word_embedding(captions)  # (batch_size, seq_len, embed_size)
        caption_embeddings = self.positional_encoding(caption_embeddings)
        caption_embeddings = self.dropout(caption_embeddings)
        
        # Create causal mask for decoder
        seq_len = captions.size(1)
        causal_mask = self.generate_causal_mask(seq_len).to(captions.device)
        
        # Transformer decoder forward pass
        output = self.transformer_decoder(
            tgt=caption_embeddings,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )
        
        # Generate vocabulary scores
        output = self.output_projection(output)  # (batch_size, seq_len, vocab_size)
        
        return output
    
    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for transformer decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def sample(self, features: torch.Tensor, max_len: int = 20) -> list:
        """Generate caption using greedy search"""
        self.eval()
        sampled_ids = []
        
        with torch.no_grad():
            batch_size = features.size(0)
            memory = self.feature_projection(features.unsqueeze(1))
            
            # Start with <start> token (assuming 1 is <start>)
            input_ids = torch.ones(batch_size, 1, dtype=torch.long, device=features.device)
            
            for i in range(max_len):
                # Embed current tokens
                embeddings = self.word_embedding(input_ids)
                embeddings = self.positional_encoding(embeddings)
                
                # Generate causal mask
                seq_len = input_ids.size(1)
                causal_mask = self.generate_causal_mask(seq_len).to(features.device)
                
                # Transformer forward pass
                output = self.transformer_decoder(
                    tgt=embeddings,
                    memory=memory,
                    tgt_mask=causal_mask
                )
                
                # Generate next token
                logits = self.output_projection(output[:, -1, :])  # (batch_size, vocab_size)
                _, predicted = logits.max(1)  # (batch_size,)
                
                sampled_ids.append(predicted.item())
                
                # Stop if <end> token is generated
                if predicted.item() == 2:  # Assuming 2 is <end> token
                    break
                
                # Append predicted token
                input_ids = torch.cat([input_ids, predicted.unsqueeze(1)], dim=1)
        
        return sampled_ids


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, embed_size: int, max_seq_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * 
                           (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model"""
    
    def __init__(self, encoder_type: str = 'resnet', decoder_type: str = 'lstm',
                 embed_size: int = 256, hidden_size: int = 512, 
                 vocab_size: int = 1000, fine_tune_encoder: bool = False):
        super(ImageCaptioningModel, self).__init__()
        
        # CNN Encoder
        self.encoder = CNNEncoder(embed_size, fine_tune_encoder)
        
        # Decoder
        if decoder_type == 'lstm':
            self.decoder = RNNDecoder(embed_size, hidden_size, vocab_size)
        elif decoder_type == 'transformer':
            self.decoder = TransformerDecoder(embed_size, hidden_size, vocab_size)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
        self.decoder_type = decoder_type
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor, 
                lengths: Optional[list] = None, padding_mask: Optional[torch.Tensor] = None):
        """Forward pass through the complete model"""
        features = self.encoder(images)
        
        if self.decoder_type == 'lstm':
            return self.decoder(features, captions, lengths)
        else:  # transformer
            return self.decoder(features, captions, padding_mask)
    
    def generate_caption(self, image: torch.Tensor, max_len: int = 20) -> list:
        """Generate caption for a single image"""
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            
            if self.decoder_type == 'lstm':
                return self.decoder.sample(features, max_len=max_len)
            else:  # transformer
                return self.decoder.sample(features, max_len=max_len)
