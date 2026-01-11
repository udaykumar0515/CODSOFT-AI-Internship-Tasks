"""
Image Captioning Demo - Task 3
CODSOFT AI Internship

Simple demo that combines:
- CNN (ResNet50) for image feature extraction
- LSTM for caption generation
"""

import os
from inference import ImageCaptioner


def main():
    """Main function for image captioning demo"""
    print("=" * 60)
    print("IMAGE CAPTIONING - CODSOFT AI INTERNSHIP - TASK 3")
    print("=" * 60)
    print("\nWelcome to the Image Captioning System!")
    print("Using: ResNet50 (CNN) + LSTM (RNN) for caption generation")
    print("=" * 60)
    
    # Initialize the captioner
    print("\nInitializing model...")
    try:
        captioner = ImageCaptioner()
        print("Model ready!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Check for sample images
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\nPlease add sample images to '{sample_dir}' directory")
        return
    
    # Get sample images
    image_files = []
    for filename in os.listdir(sample_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(os.path.join(sample_dir, filename))
    
    if not image_files:
        print(f"\nNo images found in '{sample_dir}'")
        print("Please add some .jpg, .png, or .bmp images to test")
        return
    
    print(f"\nFound {len(image_files)} images:")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {os.path.basename(img)}")
    
    # Generate captions
    print("\nGenerating captions...")
    print("-" * 60)
    
    try:
        captions = captioner.caption_multiple(image_files)
        
        print("\nGENERATED CAPTIONS:")
        print("=" * 60)
        for img_path, caption in zip(image_files, captions):
            print(f"\n{os.path.basename(img_path)}:")
            print(f"  -> {caption}")
        
        # Save results
        output_file = "captions_output.txt"
        with open(output_file, 'w') as f:
            f.write("Image Captioning Results\n")
            f.write("=" * 40 + "\n\n")
            for img_path, caption in zip(image_files, captions):
                f.write(f"{os.path.basename(img_path)}: {caption}\n")
        
        print(f"\n{'-' * 60}")
        print(f"Results saved to: {output_file}")
        print("=" * 60)
        print("\nDone! The captions are generated using:")
        print("  - ResNet50 (pre-trained CNN) for image features")
        print("  - LSTM (RNN) for caption generation")
        print("\nNote: Model uses random weights for demonstration.")
        print("For better results, train on datasets like Flickr8k or COCO.")
        
    except Exception as e:
        print(f"Error generating captions: {e}")


if __name__ == "__main__":
    main()
