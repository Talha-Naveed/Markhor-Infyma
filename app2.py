import torch
import streamlit as st
from transformers import ViTForImageClassification
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load the trained model

model_path = 'Model/model.pth' # Change it accordingly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=5, ignore_mismatched_sizes=True)
model.classifier = torch.nn.Linear(model.config.hidden_size, 5)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("Image Classification with ViT")
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_image = transform(image).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(input_image).logits
        prediction = output.argmax(1).item()

    # Display prediction
    st.write(f"Predicted Class: {prediction}")
