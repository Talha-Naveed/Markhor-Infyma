import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os

# Set up Streamlit page with vibrant colors
st.set_page_config(page_title="Retinopathy Classification", layout="wide")
st.markdown("""
    <style>
        body {background-color: #f0f8ff;}
        .stTitle {color: #ff4500; font-size: 32px; font-weight: bold;}
        .stSidebar {background-color: #ffebcd; padding: 10px;}
        .stButton>button {background-color: #ff6347; color: white; font-size: 16px;}
        .stProgress>div>div {background-color: #32cd32;}
    </style>
""", unsafe_allow_html=True)

st.title("🌟 Retinopathy Image Classification using Vision Transformer (ViT) 🌟")
st.write("Upload a retinal image to classify the stage of diabetic retinopathy.")

# Sidebar settings with vibrant colors
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("🔍 Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Upload model
uploaded_model = st.sidebar.file_uploader("📥 Upload a PyTorch model (.pth or .pt)", type=["pth", "pt"])


@st.cache_resource
def load_model(model_path=None):
    if model_path:
        model = models.vit_b_16(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()
    return model


# Load class labels
@st.cache_resource
def load_labels():
    return ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# Load model and labels
try:
    model_path = None
    if uploaded_model is not None:
        model_path = os.path.join("model.pth", uploaded_model.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())

    model = load_model(model_path)
    class_names = load_labels()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    with st.spinner("🔄 Classifying..."):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_index = torch.max(probabilities, 0)

        class_name = class_names[top_index.item()]
        confidence = top_prob.item()

        if confidence >= confidence_threshold:
            st.write(
                f"### 🎯 Prediction: <span style='color:#ff4500;'>{class_name}</span> ({confidence:.2%} confidence)",
                unsafe_allow_html=True)
            st.progress(confidence)
        else:
            st.warning("⚠️ No confident prediction above threshold.")


