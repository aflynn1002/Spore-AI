import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import os

class FungusClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # Create the same model architecture used during training
        self.model = models.resnet50(pretrained=False)
        
        # Set the number of classes (update this to match your training)
        num_fungi = 34  # For example, 2 classes: update as needed
        self.model.fc = nn.Linear(self.model.fc.in_features, num_fungi)
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # Define image transformations (same as during training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        # List of class names in the same order as your training dataset
        self.class_names = [
           "Agaricus augustus",
           "Amanita calyptroderma",
           "Amanita muscaria",
           "Amanita pantherina",
           "Amanita phalloides",
           "Amanita velosa",
           "Ascocoryne sarcoides",
           "Boletus edulis",
           "Boletus satanas",
           "Cantharellus californicus",
           "Cantharellus cibarius",
           "Chroogomphus vinicolor",
           "Clathrus ruber",
           "Clitocybe nuda",
           "Coprinus comatus",
           "Craterellus cornucopioides",
           "Ganoderma applanatum",
           "Geastrales",
           "Gliophorus psittacinus",
           "Helvella lacunosa",
           "Hericium erinaceus",
           "Hygrocype coccinea",
           "Lactarius rubidus",
           "Laetiporus sulphureus",
           "Marasmiellus candidus",
           "Morchella eximia",
           "Morchella sextelata",
           "Omphalotus olivascens",
            "Pleurotus ostreatus",
            "Sarcoscypha coccinea",
            "Stropharia ambigua",
            "Suillus fuscotomentosus",
            "Trametes versicolor",
            "Tremella mesenterica",
        ]

        self.class_names_ohio = [
            "Artomyces pyxidatus",
            "Coprineus micaceus",
            "Gyromitra esculenta",
            "Irpex lacteus",
            "Kretzschmaria deusta",
            "Megacollybia rodmani",
            "Morchella angusticeps",
            "Morchella esculentoides",
            "Morchella punctipes",
            "Neofavolus alveolaris",
            "Phellinus gilvus",
            ""

        ]
    
    def predict(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("Input should be a PIL Image")

        image = self.transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        return self.class_names[class_idx]
