import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ViT
import torch.nn.functional as F

# Page config
st.set_page_config(
    page_title="Waste Classification App",
    page_icon="♻️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .prediction {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(
        image_size=224,
        hidden_size=64,
        num_hidden_layers=3,  # Changed from 4 to 3
        intermediate_size=4*64,
        num_classes=2,  # Changed from 5 to 2
        num_attention_heads=5,  # Changed from 4 to 5
    ).to(device)

    try:
        model.load_state_dict(torch.load(r'D:\python_code\VIT\model_ 80.pt', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model_80.pt' exists in the correct path.")
        return None, device

class_names = ['CLOTHES', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

def get_prediction(model, device, image):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Transform and predict
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = F.softmax(outputs, dim=1)
        probs, predicted = torch.max(probabilities, 1)
        return predicted.item(), probs.item(), probabilities.squeeze().tolist()


def main():
    st.markdown('<p class="big-font">♻️ Garbage Classification App</p>', unsafe_allow_html=True)
    st.write("Upload an image to classify the type of waste")

    # Load model
    model, device = load_model()

    if model is None:
        return

    # Define class labels and their descriptions
    class_info = {
        'CLOTHES': {
            'description': 'Textile waste including clothing, fabrics, and other cloth materials.',
            'color': '#FF9999'
        },
        'GLASS': {
            'description': 'Glass materials such as bottles, jars, and broken glass pieces.',
            'color': '#99FF99'
        },
        'METAL': {
            'description': 'Metal waste including cans, foils, and other metallic objects.',
            'color': '#9999FF'
        },
        'PAPER': {
            'description': 'Paper products including cardboard, newspapers, and paper packaging.',
            'color': '#FFFF99'
        },
        'PLASTIC': {
            'description': 'Plastic materials such as bottles, containers, and plastic packaging.',
            'color': '#FF99FF'
        }
    }
    class_names = list(class_info.keys())

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Open and display image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Make prediction
                predicted_idx, confidence, all_probs = get_prediction(model, device, image)
                predicted_class = class_names[predicted_idx]

                # Display prediction
                st.markdown(
                    f'<div class="prediction" style="background-color: {class_info[predicted_class]["color"]}80;">'
                    f'<h3>Prediction: {predicted_class}</h3>'
                    f'<p>Confidence: {confidence * 100:.2f}%</p>'
                    f'<p>{class_info[predicted_class]["description"]}</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    # Display class information
    if uploaded_file is None:
        st.write("### Waste Categories")
        for class_name, info in class_info.items():
            st.markdown(
                f'<div style="background-color: {info["color"]}80; padding: 10px; margin: 5px; border-radius: 5px;">'
                f'<h4>{class_name}</h4>'
                f'<p>{info["description"]}</p>'
                '</div>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()