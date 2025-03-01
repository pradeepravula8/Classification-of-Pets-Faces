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
