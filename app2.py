import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import requests
import os
import numpy as np
from io import BytesIO

# Page setup
st.set_page_config(page_title="Diabetic Retinopathy Classification", layout="wide")

# Custom CSS for medical-themed UI
st.markdown("""
<style>
    .main-title { color: #2C3E50; font-size: 42px !important; font-weight: bold; text-align: center; }
    .subtitle { color: #3498DB; font-size: 22px !important; font-weight: 500; text-align: center; }
    .section-header { color: #2980B9; font-size: 24px !important; font-weight: bold; border-bottom: 2px solid #2980B9; padding-bottom: 5px; }
    .success-message { background-color: #D1FFD1; padding: 10px; border-left: 5px solid #00CC00; }
    .warning-message { background-color: #FFF4D1; padding: 10px; border-left: 5px solid #FFD000; }
    .instruction-box { background-color: #EBF5FB; padding: 15px; border-left: 5px solid #3498DB; border-radius: 10px; }
    .severity-0 { background-color: #D5F5E3; padding: 8px; border-radius: 5px; }
    .severity-1 { background-color: #FCF3CF; padding: 8px; border-radius: 5px; }
    .severity-2 { background-color: #FAE5D3; padding: 8px; border-radius: 5px; }
    .severity-3 { background-color: #F5CBA7; padding: 8px; border-radius: 5px; }
    .severity-4 { background-color: #F5B7B1; padding: 8px; border-radius: 5px; }
    .medical-info { background-color: #EBF5FB; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# Display Titles
st.markdown('<p class="main-title">Diabetic Retinopathy Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Retinal Image Analysis</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Settings")

# Define retinopathy class names and descriptions
retinopathy_classes = [
    "No DR (0)",
    "Mild DR (1)",
    "Moderate DR (2)",
    "Severe DR (3)",
    "Proliferative DR (4)"
]

retinopathy_descriptions = {
    "No DR (0)": "No visible signs of diabetic retinopathy.",
    "Mild DR (1)": "Microaneurysms only - small, round dots of blood that leak into the retina.",
    "Moderate DR (2)": "More microaneurysms, plus some retinal hemorrhages or hard exudates.",
    "Severe DR (3)": "Many hemorrhages, venous beading, or prominent intraretinal microvascular abnormalities.",
    "Proliferative DR (4)": "Advanced disease with growth of new blood vessels and/or risk of retinal detachment."
}


@st.cache_resource
def load_model_from_github():
    """Load a model from GitHub releases."""
    model_url = st.session_state.get('model_url', "")
    if not model_url:
        return None, False

    model_filename = "retinopathy_model.pth"
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    model_path = os.path.join(temp_dir, model_filename)

    try:
        with st.spinner("Downloading retinopathy model from GitHub..."):
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Load the PyTorch model
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()
            return model, True
    except Exception as e:
        st.error(f"⚠️ GitHub model loading failed: {e}")
        return None, False


# Image Preprocessing for Retinopathy
def preprocess_retina_image(image):
    # Resize to standard dimensions (often 224x224 or 299x299 for medical imaging)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # Normalize using ImageNet means and stds as a baseline
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# GitHub Model Settings
st.sidebar.subheader("GitHub Model Settings")

# Initialize session state for URL if it doesn't exist
if 'model_url' not in st.session_state:
    st.session_state.model_url = "https://github.com/Talha-Naveed/Infyma/releases/download/0.2/model.pth"

model_url = st.sidebar.text_input("Model URL (GitHub Release)", value=st.session_state.model_url)

# Update session state when URL changes
if model_url != st.session_state.model_url:
    st.session_state.model_url = model_url

# Load model from GitHub
model, model_loaded = load_model_from_github()

# Model Loading Status
if model_loaded:
    st.markdown('<div class="success-message">✅ Retinopathy classification model loaded successfully!</div>',
                unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="warning-message">⚠️ No retinopathy model loaded. Please ensure you have provided a valid GitHub URL.</div>',
        unsafe_allow_html=True)

# Confidence Threshold
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Add medical disclaimer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="medical-info">
<strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper diagnosis and treatment of diabetic retinopathy.
</div>
""", unsafe_allow_html=True)

# File Uploader
st.markdown('<p class="section-header">Upload Retinal Image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a retinal fundus image", type=["jpg", "jpeg", "png"])

# Display image enhancement options
if uploaded_file is not None:
    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Load and display original image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded Retinal Image", width=None, use_container_width=True)

        # Image enhancement options
        enhance_image = st.checkbox("Apply image enhancement", value=True)

        if enhance_image:
            # Create a copy of the image for enhancement
            enhanced_image = image.copy()

            # Simple contrast enhancement for better visualization
            enhance_contrast = st.slider("Enhance contrast", 1.0, 3.0, 1.5, 0.1)
            enhance_brightness = st.slider("Enhance brightness", 0.7, 1.3, 1.0, 0.05)

            # Apply enhancements using PIL
            from PIL import ImageEnhance

            # Adjust contrast
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(enhance_contrast)

            # Adjust brightness
            enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = enhancer.enhance(enhance_brightness)

            # Show enhanced image
            st.image(enhanced_image, caption="Enhanced Image", width=None, use_container_width=True)

            # Use enhanced image for prediction
            image_for_prediction = enhanced_image
        else:
            # Use original image for prediction
            image_for_prediction = image

    # Predict button
    predict_button = st.button("Analyze Retinal Image", use_container_width=True, disabled=not model_loaded)

    if predict_button:
        with col2:
            with st.spinner("Analyzing retinal image for diabetic retinopathy..."):
                # Preprocess image
                input_tensor = preprocess_retina_image(image_for_prediction)

                try:
                    with torch.no_grad():
                        # Get model predictions
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)

                    # Get the predicted class and confidence
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_class].item()

                    # Display results header
                    st.markdown('<p class="section-header">Analysis Results</p>', unsafe_allow_html=True)

                    # Display the predicted severity
                    st.markdown(
                        f'<div class="severity-{pred_class}"><h3>Predicted: {retinopathy_classes[pred_class]}</h3></div>',
                        unsafe_allow_html=True)
                    st.write(f"**Confidence**: {confidence:.1%}")

                    # Description of the severity level
                    st.markdown("### Description")
                    st.write(retinopathy_descriptions[retinopathy_classes[pred_class]])

                    # Show recommendations based on DR severity
                    st.markdown("### Recommended Actions")
                    if pred_class == 0:
                        st.write("• Maintain regular diabetes management")
                        st.write("• Annual eye screening recommended")
                    elif pred_class == 1:
                        st.write("• Monitor blood sugar levels closely")
                        st.write("• Follow-up eye examination in 6-12 months")
                    elif pred_class == 2:
                        st.write("• Consult an ophthalmologist")
                        st.write("• Follow-up eye examination in 3-6 months")
                    elif pred_class == 3:
                        st.write("• Prompt referral to an ophthalmologist or retina specialist")
                        st.write("• Follow-up examination in 1-3 months")
                    else:  # Proliferative DR
                        st.write("• Urgent referral to a retina specialist")
                        st.write("• May require immediate treatment (e.g., laser therapy)")

                    # Show detailed results in expander
                    with st.expander("View Detailed Analysis"):
                        # Show all class probabilities
                        st.markdown("#### Severity Level Probabilities")
                        results = []
                        for i, class_name in enumerate(retinopathy_classes):
                            prob = probabilities[i].item()
                            results.append({
                                "Severity Level": class_name,
                                "Probability": f"{prob:.4f}"
                            })

                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)

                        # Create bar chart for visualization
                        chart_data = pd.DataFrame({
                            'Severity': [c.split(' ')[0] for c in retinopathy_classes],
                            'Probability': [probabilities[i].item() for i in range(len(retinopathy_classes))]
                        })

                        st.bar_chart(chart_data.set_index('Severity'))

                except Exception as e:
                    st.error(f"❌ Error during retinal analysis: {e}")
                    st.info("Please make sure you've uploaded a clear retinal fundus image.")

# Instructions
st.markdown('<p class="section-header">How to Use This Diabetic Retinopathy Classifier</p>', unsafe_allow_html=True)
st.markdown('<div class="instruction-box">', unsafe_allow_html=True)

st.markdown('<p><strong>Step 1:</strong> Set up the retinopathy classification model</p>', unsafe_allow_html=True)
st.markdown('• Enter the URL to your retinopathy model in GitHub releases.', unsafe_allow_html=True)
st.markdown('• The default model URL points to a pre-trained diabetic retinopathy classifier.', unsafe_allow_html=True)

st.markdown('<p><strong>Step 2:</strong> Upload a retinal fundus image</p>', unsafe_allow_html=True)
st.markdown('• Use the file uploader to select a high-quality image of the retina', unsafe_allow_html=True)
st.markdown('• The image should be centered on the macula and include the optic disc', unsafe_allow_html=True)

st.markdown('<p><strong>Step 3:</strong> Adjust image enhancement if needed</p>', unsafe_allow_html=True)
st.markdown('• Use the sliders to improve visibility of retinal features', unsafe_allow_html=True)

st.markdown('<p><strong>Step 4:</strong> Click "Analyze Retinal Image" to classify</p>', unsafe_allow_html=True)
st.markdown('• View the results and recommended actions based on severity', unsafe_allow_html=True)
st.markdown('• Expand "View Detailed Analysis" to see probabilities for all severity levels', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Educational information
with st.expander("Learn About Diabetic Retinopathy"):
    st.markdown("""
    ### What is Diabetic Retinopathy?

    Diabetic retinopathy (DR) is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels in the retina (the light-sensitive tissue at the back of the eye).

    ### The Five Stages of Diabetic Retinopathy

    1. **No DR (Stage 0)**: No visible abnormalities
    2. **Mild NPDR (Stage 1)**: Microaneurysms only
    3. **Moderate NPDR (Stage 2)**: Microaneurysms, dot and blot hemorrhages, hard exudates
    4. **Severe NPDR (Stage 3)**: Numerous hemorrhages, venous beading, intraretinal microvascular abnormalities (IRMA)
    5. **Proliferative DR (Stage 4)**: Growth of new abnormal blood vessels, potential retinal detachment

    ### Risk Factors

    - Duration of diabetes
    - Poor blood sugar control
    - High blood pressure
    - High cholesterol
    - Pregnancy
    - Tobacco use

    ### Prevention and Management

    - Regular eye examinations
    - Managing blood sugar levels
    - Controlling blood pressure and cholesterol
    - Regular exercise and healthy diet
    - Not smoking
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; color: #888;">
    <hr style="width:50%; margin:20px auto;">
    <p>Diabetic Retinopathy Classification | AI-powered retinal image analysis</p>
</div>
""", unsafe_allow_html=True)
