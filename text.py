import os
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

# Make sure we're using CPU if CUDA is not available
device = "cpu"
print(f"Using device: {device}")

class PetDataset:
    def __init__(self, data_dir):
        """
        Initialize the Pet Dataset
        Args:
            data_dir (str): Directory containing the pet images organized by emotions
        """
        self.data_dir = Path(data_dir)
        
        # Verify directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found at {self.data_dir}")
        
        # Get all emotion folders
        self.emotion_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        print(f"Found emotion folders: {[f.name for f in self.emotion_folders]}")
        
        # Initialize the model
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.model.to(device)
        
        # Setup image transformation
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
            
    def classify_pet(self, image):
        """
        Classify the type of pet in the image
        Args:
            image: PIL Image object
        Returns:
            str: predicted pet type
        """
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
        
        # Get class name from ImageNet labels
        class_name = ResNet50_Weights.DEFAULT.meta["categories"][predicted_idx.item()]
        
        # Filter for pet-related classes
        pet_keywords = ['cat', 'dog', 'rabbit', 'hamster', 'cow', 'horse']
        for keyword in pet_keywords:
            if keyword in class_name.lower():
                return keyword.capitalize()
        return "Other"
            
    def load_dataset(self):
        """
        Load the dataset and create a DataFrame with image paths, emotions, and pet types
        Returns:
            pandas DataFrame with columns: image_path, emotion, pet_type
        """
        data = []

 for emotion_folder in self.emotion_folders:
            emotion = emotion_folder.name
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files = list(emotion_folder.glob(f'*{ext}')) + list(emotion_folder.glob(f'*{ext.upper()}'))
                
                for img_path in image_files:
                    try:
                        # Load and classify image
                        image = self.load_image(img_path)
                        pet_type = self.classify_pet(image)
                        
                        data.append({
                            'image_path': str(img_path),
                            'emotion': emotion,
                            'pet_type': pet_type
                        })
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        df = pd.DataFrame(data)
        print(f"\nFound {len(df)} total images:")
        print("\nPet type distribution:")
        print(df['pet_type'].value_counts())
        print("\nEmotion distribution:")
        for emotion in df['emotion'].unique():
            count = len(df[df['emotion'] == emotion])
            print(f"{emotion}: {count} images")
            
        return df
    
    def load_image(self, image_path):
        """
        Load and return an image
        Args:
            image_path (str): Path to the image file
        Returns:
            PIL Image object
        """
        return Image.open(image_path).convert('RGB')

def main():
    dataset_path = r"C:\Users\pradeep\Downloads\pets_faces"
    
    try:
        print("Initializing pet dataset and loading model...")
        pet_dataset = PetDataset(dataset_path)
        
        print("\nProcessing images...")
        df = pet_dataset.load_dataset()
        
        print(f"\nTotal number of images: {len(df)}")
        print("\nPet type and emotion distribution:")
        print(pd.crosstab(df['pet_type'], df['emotion']))
        
        # Example: Load and analyze first image
        if len(df) > 0:
            first_image = pet_dataset.load_image(df.iloc[0]['image_path'])
            first_emotion = df.iloc[0]['emotion']
            first_pet_type = df.iloc[0]['pet_type']
            print(f"\nFirst image details:")
            print(f"Pet Type: {first_pet_type}")
            print(f"Emotion: {first_emotion}")
            print(f"Size: {first_image.size}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

