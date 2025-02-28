# Classification-of-Pets-Faces

Overview

This project classifies pet images into different categories based on emotions and identifies the type of pet in the image using a pre-trained ResNet50 model. The dataset is structured with images stored in folders named after different emotions

Features

Uses ResNet50 to classify pet types from images.

Processes images from directories named after emotions.

Generates a pandas DataFrame containing image paths, emotions, and predicted pet types.

Provides summary statistics on pet types and emotions.

Installation & Requirements

Dependencies

Ensure you have the following Python libraries installed:

pip install torch torchvision pandas pillow

Directory Structure

Your dataset should be organized as follows:

/pets_faces/
    /happy/
        cat1.jpg
        dog2.png
    /sad/
        rabbit1.jpg
        cat3.jpeg
    ...

Usage
Set the dataset path in the script:

dataset_path = r"C:\Users\pradeep\Downloads\pets_faces"

Run the script:

python pet_classifier.py

The script will:

Load images from the dataset directory.

Classify pet types using ResNet50.

Output a summary of pet type distribution per emotion.


Notes

The script filters images by checking for pet-related categories in the ResNet50 classification output.

If an image does not belong to known pet classes (dog, cat, rabbit, etc.), it is labeled as "Other".

Ensure that your dataset is correctly structured for proper classification results.
