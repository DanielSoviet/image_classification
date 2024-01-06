!pip -q install streamlit

from google.colab import drive
drive.mount('/content/drive')

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the trained TensorFlow model
model_path = "/content/drive/MyDrive/models/keras_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the labels
class_labels = open("/content/drive/MyDrive/models/labels.txt", "r").readlines()
st.title("Image Classification Animals")
st.header("Animals: Dog, Horse, Elephant, Butterfly, Chicken, Cat, Cow, Sheep, Spider, Squirrel.")
st.write("Upload your Image for image classification")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = np.array(image.resize((224, 224))) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")  # Display confidence with 2 decimal places

!streamlit run image.py & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
