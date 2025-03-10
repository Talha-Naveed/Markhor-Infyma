import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
import logging
from io import BytesIO
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DR_Classifier")

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
    .error-message { background-color: #FFE5E5; padding: 10px; border-left: 5px solid #FF0000; }
    .info-message { background-color: #E5F6FF; padding: 10px; border-left: 5px solid #0088FF; }
    .instruction-box { background-color: #EBF5FB; padding: 15px; border-left: 5px solid #3498DB; border-radius: 10px; }
    .severity-0 { background-color: #D5F5E3; padding: 8px; border-radius: 5px; }
    .severity-1 { background-color: #FCF3CF; padding: 8px; border-radius: 5px; }
    .severity-2 { background-color: #FAE5D3; padding: 8px; border-radius: 5px; }
    .severity-3 { background-color: #F5CBA7; padding: 8px; border-radius: 5px; }
    .severity-4 { background-color: #F5B7B1; padding: 8px; border-radius: 5px; }
    .medical-info { background-color: #EBF5FB; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 14px; }
    .log-container { background-color: #F8F9F9; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
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


# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_path' not in st.session_state:
    st.session_state.model_path = "model.pth"
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []


# Function to add log message to session state
def add_log(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {level} - {message}"
    st.session_state.log_messages.append(log_entry)

    # Also log to the file
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        logger.debug(message)


# Function to verify model is valid for inference
def verify_model(model):
    try:
        # Check if model has expected methods
        if not hasattr(model, 'eval'):
            add_log("Model doesn't have 'eval' method - not a PyTorch model", "ERROR")
            return False

        # Set to evaluation mode
        model.eval()

        # Create a dummy input tensor of expected size
        dummy_input = torch.zeros(1, 3, 299, 299)

        # Try to get output dimensions - this will check if model is runnable
        with torch.no_grad():
            output = model(dummy_input)

        # Check if output shape matches expected number of classes
        if output.shape[1] != len(retinopathy_classes):
            add_log(
                f"Model output doesn't match expected number of classes. Got {output.shape[1]}, expected {len(retinopathy_classes)}",
                "WARNING")
            # We'll return True anyway since the model runs, but log this as a warning

        add_log("Model verification successful - ready for inference")
        return True

    except Exception as e:
        add_log(f"Model verification failed: {str(e)}", "ERROR")
        return False


# Local Model Settings
st.sidebar.subheader("Local Model Settings")

# Local model path input
model_path = st.sidebar.text_input("Model File Path", value=st.session_state.model_path)


# Function to load model from local path
def load_local_model(path):
    """Load a model from local file system."""
    if not path:
        add_log("No model path provided", "WARNING")
        return None, False

    try:
        add_log(f"Attempting to load model from: {path}")

        # Check if file exists
        if not os.path.exists(path):
            add_log(f"Model file not found at: {path}", "ERROR")
            return None, False

        # Log file size
        file_size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
        add_log(f"Model file size: {file_size:.2f} MB")

        # Load the PyTorch model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        add_log(f"Loading model using device: {device}")

        model = torch.load(path, map_location=device)
        add_log("Model loaded into memory")

        # Verify the model
        if verify_model(model):
            add_log("Model successfully loaded and verified")
            return model, True
        else:
            add_log("Model loaded but failed verification", "ERROR")
            return None, False

    except Exception as e:
        add_log(f"Error loading model: {str(e)}", "ERROR")
        return None, False


# Check if path changed and load model
if model_path != st.session_state.model_path or st.sidebar.button("Load Model"):
    st.session_state.model_path = model_path
    with st.spinner("Loading and verifying model..."):
        st.session_state.model, st.session_state.model_loaded = load_local_model(model_path)

# Display debug log in expander
with st.sidebar.expander("View Debug Logs"):
    log_content = "\n".join(st.session_state.log_messages[-50:])  # Show last 50 log entries
    st.markdown(f'<div class="log-container">{log_content}</div>', unsafe_allow_html=True)
    if st.button("Clear Logs"):
        st.session_state.log_messages = []

# Model Loading Status
if st.session_state.model_loaded:
    st.markdown('<div class="success-message">✅ Retinopathy classification model loaded successfully!</div>',
                unsafe_allow_html=True)

    # Show model details
    with st.expander("Model Information"):
        # Attempt to get model architecture
        try:
            model_info = str(st.session_state.model)
            st.code(model_info, language="text")

            # Count parameters
            total_params = sum(p.numel() for p in st.session_state.model.parameters())
            trainable_params = sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad)

            st.write(f"**Total parameters:** {total_params:,}")
            st.write(f"**Trainable parameters:** {trainable_params:,}")

        except Exception as e:
            st.write(f"Could not extract model details: {e}")
else:
    st.markdown(
        '<div class="warning-message">⚠️ No retinopathy model loaded. Please ensure you have provided a valid local path and click "Load Model".</div>',
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

        add_log(f"Image uploaded: {uploaded_file.name}, Size: {image.size}")

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

            add_log(f"Image enhanced: Contrast={enhance_contrast}, Brightness={enhance_brightness}")

            # Use enhanced image for prediction
            image_for_prediction = enhanced_image
        else:
            # Use original image for prediction
            image_for_prediction = image
            add_log("Using original image without enhancement")

    # Predict button
    predict_button = st.button("Analyze Retinal Image", use_container_width=True,
                               disabled=not st.session_state.model_loaded)

    if predict_button:
        with col2:
            with st.spinner("Analyzing retinal image for diabetic retinopathy..."):
                add_log("Starting image analysis...")

                # Preprocess image
                try:
                    input_tensor = preprocess_retina_image(image_for_prediction)
                    add_log(f"Image preprocessed to tensor shape: {input_tensor.shape}")
                except Exception as e:
                    add_log(f"Error preprocessing image: {str(e)}", "ERROR")
                    st.error(f"Error preprocessing image: {str(e)}")

                try:
                    start_time = time.time()
                    with torch.no_grad():
                        # Get model predictions
                        output = st.session_state.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)

                    inference_time = (time.time() - start_time) * 1000  # ms
                    add_log(f"Inference completed in {inference_time:.2f} ms")

                    # Get the predicted class and confidence
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_class].item()

                    add_log(
                        f"Prediction result: Class={pred_class} ({retinopathy_classes[pred_class]}), Confidence={confidence:.4f}")

                    # Display results header
                    st.markdown('<p class="section-header">Analysis Results</p>', unsafe_allow_html=True)

                    # Display the predicted severity
                    st.markdown(
                        f'<div class="severity-{pred_class}"><h3>Predicted: {retinopathy_classes[pred_class]}</h3></div>',
                        unsafe_allow_html=True)
                    st.write(f"**Confidence**: {confidence:.1%}")
                    st.write(f"**Inference Time**: {inference_time:.2f} ms")

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
                    error_msg = str(e)
                    add_log(f"Error during retinal analysis: {error_msg}", "ERROR")
                    st.error(f"❌ Error during retinal analysis: {error_msg}")
                    st.info(
                        "Please make sure you've uploaded a clear retinal fundus image and that the model is compatible.")

# Instructions
st.markdown('<p class="section-header">How to Use This Diabetic Retinopathy Classifier</p>', unsafe_allow_html=True)
st.markdown('<div class="instruction-box">', unsafe_allow_html=True)

st.markdown('<p><strong>Step 1:</strong> Set up the retinopathy classification model</p>', unsafe_allow_html=True)
st.markdown('• Enter the path to your local retinopathy model file.', unsafe_allow_html=True)
st.markdown('• Click the "Load Model" button to load the model.', unsafe_allow_html=True)
st.markdown('• Check the "View Debug Logs" to verify model loading success.', unsafe_allow_html=True)
st.markdown('• The default path points to "model.pth".', unsafe_allow_html=True)

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

# Initialize the app log
if len(st.session_state.log_messages) == 0:
    add_log("Application started")
