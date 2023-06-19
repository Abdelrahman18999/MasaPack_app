import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Title of the application
st.title("Masa Pack Web Application")

# Markdown
st.markdown("This is a web application service from Masa Pack company to make product inspection")

# Header
st.header("Upload The Product Image")

# Upload an image file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Classification Function
def classify(image, model, classNames):
    target_size = 224

    resized_image = cv2.resize(image, (target_size, target_size))
    cropped_image = resized_image[40:140, 50:150]
    scaled_image = cropped_image.astype("float32") / 255.0

    # convert the image to a numpy array
    img_array = np.array(scaled_image)

    # expand the dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    # make a prediction using the loaded model
    prediction = model.predict(img_array)
    print(prediction[0])

    if prediction[0] >= 0.5:
        index = 1
        confidence_score = prediction[0]
    else:
        index = 0
        confidence_score = 1 - prediction[0]
        
    classname = classNames[index]

    return classname, confidence_score

# Display image and perform classification
# Load the saved model
model = load_model('my_model.h5')
if file is not None:
    classNames = ['Flawless', 'Defect']
    image = Image.open(file).convert('L')  # Read the image in grayscale
    image_cv = np.array(image)

    st.image(image, use_column_width=True)

    # Classify image
    classname, conf_score = classify(image_cv, model, classNames)

    # Write classification
    st.write("## {}".format(classname))
    st.write("### score: {}%".format(int(conf_score * 100)))
