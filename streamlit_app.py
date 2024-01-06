$ pip install tensorflow
$ pip install streamlit
$ pip install numpy

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model_path = "keras_model.h5"
model = tf.keras.models.load_model(model_path)

class_labels = open("labels.txt", "r").readlines()
st.title("Image Classification Animals")
st.header("Animals: Dog, Horse, Elephant, Butterfly, Chicken, Cat, Cow, Sheep, Spider, Squirrel.")
st.write("Upload your Image for image classification")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = np.array(image.resize((224, 224))) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
