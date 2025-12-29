#!/usr/bin/env python3
"""
Image Captioning Main Script - Task 3
CODSOFT AI Internship

Main entry point for image captioning system.
"""

import os
import sys
from inference import ImageCaptioningInference


def main():
    """Main function for image captioning demo"""
    print("=" * 60)
    print("🖼️  IMAGE CAPTIONING - CODSOFT AI INTERNSHIP")
    print("=" * 60)
    print("\nWelcome to the Image Captioning System!")
    print("This system uses CNN + RNN/Transformer to generate captions for images.")
    print("\nFeatures:")
    print("- CNN encoder (ResNet50) for feature extraction")
    print("- RNN (LSTM) or Transformer decoder for caption generation")
    print("- Support for single image or batch processing")
    print("- Preprocessing with ImageNet normalization")
    print("\n" + "=" * 60)
    
    # Initialize the model
    print("\n🤖 Initializing Image Captioning Model...")
    print("Note: Using random weights (no pretrained model available)")
    print("In a real scenario, you would load a trained model.")
    
    try:
        inference = ImageCaptioningInference(
            model_path=None,  # No pretrained model for demo
            decoder_type='lstm'  # Can be 'lstm' or 'transformer'
        )
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    # Check for sample images
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        print(f"\n📁 Creating sample images directory: {sample_dir}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Please add some sample images to the '{sample_dir}' directory")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        return
    
    # Get sample images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    sample_images = []
    
    for filename in os.listdir(sample_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            sample_images.append(os.path.join(sample_dir, filename))
    
    if not sample_images:
        print(f"\n⚠️  No sample images found in '{sample_dir}' directory")
        print("Please add some images to test the captioning system.")
        return
    
    print(f"\n📸 Found {len(sample_images)} sample images:")
    for i, img_path in enumerate(sample_images, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    
    # Process images
    print("\n🔄 Generating captions...")
    print("-" * 60)
    
    try:
        captions = inference.generate_batch_captions(sample_images, max_length=15)
        
        print("\n" + "=" * 60)
        print("🎯 GENERATED CAPTIONS")
        print("=" * 60)
        
        for img_path, caption in zip(sample_images, captions):
            img_name = os.path.basename(img_path)
            print(f"📷 {img_name}")
            print(f"💬 {caption}")
            print("-" * 40)
        
        print("\n✅ Caption generation completed!")
        
        # Save captions to file
        output_file = "generated_captions.txt"
        with open(output_file, 'w') as f:
            f.write("Image Captioning Results\n")
            f.write("=" * 40 + "\n\n")
            for img_path, caption in zip(sample_images, captions):
                f.write(f"{os.path.basename(img_path)}: {caption}\n")
        
        print(f"📄 Captions saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating captions: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Image Captioning Demo Complete!")
    print("=" * 60)
    print("\nTo use the system with your own images:")
    print("1. Place images in the 'sample_images' directory")
    print("2. Run: python main.py")
    print("\nFor advanced usage:")
    print("python inference.py --image path/to/image.jpg")
    print("python inference.py --image_dir path/to/images/")
    print("\nNote: This demo uses random weights. For real applications,")
    print("train the model on a dataset like Flickr8k or COCO Captions.")


if __name__ == "__main__":
    main()
