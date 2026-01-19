"""
MNIST Digit Classification - Interactive Demo

A Streamlit web application for real-time handwritten digit recognition.
Draw a digit and see predictions from multiple models.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LeNet, AlexNet, SimpleCNN

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .correct {
        background-color: #d4edda;
    }
    .digit-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load pre-trained models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}

    # Initialize models (in production, load pre-trained weights)
    models['LeNet'] = LeNet().to(device)
    models['AlexNet'] = AlexNet().to(device)
    models['SimpleCNN'] = SimpleCNN().to(device)

    # Set to evaluation mode
    for model in models.values():
        model.eval()

    return models, device


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess drawn image for model input.

    Args:
        image: Input image array

    Returns:
        Preprocessed tensor ready for model
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Resize to 28x28
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize
    tensor = torch.tensor(np.array(pil_image), dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    tensor = (tensor - 0.1307) / 0.3081  # MNIST normalization

    return tensor


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device):
    """
    Get prediction from a model.

    Returns:
        Tuple of (predicted_digit, confidence, all_probabilities)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted = output.argmax(dim=1).item()
        confidence = probabilities[predicted]

    return predicted, confidence, probabilities


def main():
    st.markdown('<p class="main-header">🔢 MNIST Digit Classifier</p>', unsafe_allow_html=True)

    st.markdown("""
    **Draw a digit (0-9) in the canvas below and see real-time predictions from different models!**

    This demo showcases various machine learning models trained on the MNIST dataset.
    """)

    # Load models
    models, device = load_models()

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Draw Here")

        # Drawing canvas
        try:
            from streamlit_drawable_canvas import st_canvas

            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=20,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        except ImportError:
            st.warning("Drawing canvas not available. Please install: `pip install streamlit-drawable-canvas`")
            st.info("You can upload an image instead:")
            uploaded_file = st.file_uploader("Upload digit image", type=['png', 'jpg', 'jpeg'])
            canvas_result = None

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('L')
                image = np.array(image)
                st.image(image, caption='Uploaded Image', width=280)

    with col2:
        st.subheader("🎯 Predictions")

        # Check if we have a drawing
        has_drawing = False
        image_data = None

        if canvas_result is not None and canvas_result.image_data is not None:
            image_data = canvas_result.image_data
            has_drawing = np.any(image_data[:, :, :3] > 10)

        if has_drawing and image_data is not None:
            # Preprocess the image
            image_tensor = preprocess_image(image_data[:, :, :3])

            # Get predictions from all models
            st.markdown("### Model Predictions")

            for model_name, model in models.items():
                predicted, confidence, probs = predict(model, image_tensor, device)

                col_pred, col_conf = st.columns([1, 2])
                with col_pred:
                    st.metric(
                        label=model_name,
                        value=str(predicted),
                        delta=f"{confidence*100:.1f}% confident"
                    )
                with col_conf:
                    # Mini bar chart of probabilities
                    st.bar_chart(dict(zip(range(10), probs)), height=100)

            # Show preprocessed image
            st.markdown("### Preprocessed Input (28x28)")
            processed = image_tensor.squeeze().numpy()
            processed = ((processed * 0.3081) + 0.1307) * 255
            st.image(processed.astype(np.uint8), width=140)

        else:
            st.info("👆 Draw a digit on the canvas to see predictions")

            # Show example predictions
            st.markdown("### Example Results")
            st.markdown("""
            | Model | Best Accuracy |
            |-------|--------------|
            | AlexNet | 96.85% |
            | LeNet | 96.00% |
            | SimpleCNN | 94.00% |
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Demo**

    This application demonstrates digit classification using neural networks trained on the MNIST dataset.
    The models compare different architectures from simple CNNs to more complex networks like AlexNet.

    *Note: Models in this demo use random weights. In production, load pre-trained weights for accurate predictions.*

    [View Source Code](https://github.com/thanhtrung102/Computer-Vision-group12) |
    [Documentation](https://github.com/thanhtrung102/Computer-Vision-group12/blob/main/docs/ARCHITECTURE.md)
    """)


if __name__ == "__main__":
    main()
