import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os


# https://drive.google.com/file/d/12_wt0RbN1ZQfX_PVRNdsMU8XXhbPUk8x/view?usp=drive_link



import gdown
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "cnn_model.keras"

def download_model():
    file_id = "12_wt0RbN1ZQfX_PVRNdsMU8XXhbPUk8x"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Download only if not exists
if not os.path.exists(MODEL_PATH):
    download_model()

model = load_model(MODEL_PATH)


# Load trained CNN model

# model = tf.keras.models.load_model("cnn_model1.tflite")

# Use actual class names

class_names = [
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'
]

st.title(" Plant Leaf Disease Classifier")

st.write("Upload a plant leaf image to detect disease or healthy status.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg", 'jfif', 'bmp', 'tiff', 'webp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing

    img = image.resize((224, 224))
    img_array = np.array(img) 
    img_array = np.expand_dims(img_array, axis=0)

    
    # Prediction
    
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction[0])

    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = "Unknown class"

    confidence = np.max(prediction[0]) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")