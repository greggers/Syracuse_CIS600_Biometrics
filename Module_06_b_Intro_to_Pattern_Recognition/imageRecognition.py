import torch
import cv2
import numpy as np
from torchvision import models, transforms
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable

# Load pretrained VGG16
model = models.vgg16(pretrained=True)
model.eval()

# Download a sample image (LFW face example)
import requests
url = "https://upload.wikimedia.org/wikipedia/commons/7/7c/Profile_avatar_placeholder_large.png"
resp = requests.get(url, stream=True)
with open("face.jpg", "wb") as f:
    f.write(resp.content)

# Read image with OpenCV (BGR)
image = cv2.imread("face.jpg")
# Convert BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocessing: resize, tensor, normalize
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # converts to [0,1] and shape (C,H,W)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

input_tensor = preprocess(image).unsqueeze(0)  # shape: (1, 3, 224, 224)

# Run model
with torch.no_grad():
    output = model(input_tensor)

# Get top-1 prediction
class_idx = torch.argmax(output, dim=1).item()
probability = torch.nn.functional.softmax(output, dim=1)[0, class_idx].item()

print(f"Class index: {class_idx}")
print(f"Probability: {probability:.2%}")
