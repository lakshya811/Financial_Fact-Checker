import torch
from PIL import Image
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from joblib import load
import streamlit as st

# Load models for different fusion techniques
# early_model = load(
#     'C:\\Users\\laksh\\OneDrive\\Documents\\GitHub\\Financial_Fact-Checker\\early_model.joblib')
# late_model_text = load(
#     'C:\\Users\\laksh\\OneDrive\\Documents\\GitHub\\Financial_Fact-Checker\\late_model_text.joblib')
# late_model_image = load(
#     'C:\\Users\\laksh\\OneDrive\\Documents\\GitHub\\Financial_Fact-Checker\\late_model_image.joblib')
# hybrid_model = load_model(
#     'C:\\Users\\laksh\\OneDrive\\Documents\\GitHub\\Financial_Fact-Checker\\hybrid_model.h5')

early_model = load(
    'early_model.joblib')
late_model_text = load(
    'late_model_text.joblib')
late_model_image = load(
    'late_model_image.joblib')
hybrid_model = load_model(
    'hybrid_model.h5')
# Load pretrained models for feature extraction
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
image_model = EfficientNetB0(
    weights='imagenet', include_top=False, pooling='avg')

# Preprocess and predict functions


def get_text_features(text):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        return text_model(**inputs).last_hidden_state.mean(dim=1).numpy()


def get_image_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image_model.predict(image).flatten()


def predict_early_fusion(claim, image_path):
    text_features = get_text_features(claim)
    image_features = get_image_features(image_path)

    # Ensure both have the same dimensions
    if text_features.ndim == 2 and image_features.ndim == 1:
        image_features = np.expand_dims(image_features, axis=0)

    combined_features = np.concatenate([text_features, image_features], axis=1)
    prediction = early_model.predict(combined_features)
    return prediction


def predict_late_fusion(claim, image_path):
    text_features = get_text_features(claim)
    image_features = get_image_features(image_path)

    # Ensure `text_features` and `image_features` are 2D for prediction
    # Reshape to (1, num_features) if needed
    text_features = text_features.reshape(1, -1)
    # Reshape to (1, num_features) if needed
    image_features = image_features.reshape(1, -1)

    # Make predictions with text and image models
    text_pred = late_model_text.predict(text_features)
    image_pred = late_model_image.predict(image_features)

    # Combine predictions with averaged soft voting
    return int((text_pred + image_pred) / 2 > 0.5)


def predict_hybrid_fusion(claim, image_path):
    # Get text and image features
    text_features = get_text_features(claim)
    image_features = get_image_features(image_path)

    # Ensure both features are 2D with the same number of samples (1 sample here)
    # Reshape to (1, num_text_features)
    text_features = text_features.reshape(1, -1)
    image_features = image_features.reshape(
        1, -1)  # Reshape to (1, num_image_features)

    # Make prediction with the hybrid model
    prediction = hybrid_model.predict([text_features, image_features])
    return np.argmax(prediction)


st.title("FinFact - Claim Verification")
st.write("This app verifies financial claims using multimodal fusion techniques.")

# Text input for the claim
claim_text = st.text_area("Enter the Claim Text")

# Image upload input for the claim image
image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Option to select fusion technique
fusion_technique = st.selectbox(
    "Choose a Fusion Technique",
    ("Early Fusion", "Late Fusion", "Hybrid Fusion")
)

# Show entered claim and image
if claim_text and image_file:
    # Display the claim and image
    image = Image.open(image_file)
    st.write("### Claim:")
    st.write(claim_text)
    st.write("### Image:")
    st.image(image, caption="Claim Image", use_column_width=True)

    # Predict claim's truthfulness
    if st.button("Verify Claim"):
        # Save the uploaded image temporarily for prediction
        with open("temp_image.jpg", "wb") as f:
            f.write(image_file.getbuffer())

        # Predict using the selected fusion technique
        if fusion_technique == "Early Fusion":
            prediction = predict_early_fusion(claim_text, "temp_image.jpg")
        elif fusion_technique == "Late Fusion":
            prediction = predict_late_fusion(claim_text, "temp_image.jpg")
        elif fusion_technique == "Hybrid Fusion":
            prediction = predict_hybrid_fusion(claim_text, "temp_image.jpg")

        # Display result
        result = "True" if prediction == 1 else "False" if prediction == 0 else "Not enough information"
        st.write("Prediction:", result)
else:
    st.write("Please enter a claim and upload an image to verify.")
