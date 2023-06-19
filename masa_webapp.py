import streamlit as st
# import cv2
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

    # Define the target size
    target_size = (224, 224)

    # Resize the image
    resized_image = np.resize(image, target_size)

    # Convert the cropped image to a NumPy array
    #np_image = np.array(cropped_image)

    # Normalize the image
    scaled_image = resized_image.astype("float32") / 255.0

    # expand the dimensions to match the input shape of the model
    img_array = np.expand_dims(scaled_image, axis=0)

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
    # Crop the image
    cropped_image = image.crop((50, 40, 150, 140))
    image_cv = np.array(cropped_image)

    st.image(image, use_column_width=True)

    # Classify image
    classname, conf_score = classify(image_cv, model, classNames)

    # Write classification
    st.write("## {}".format(classname))
    st.write("### score: {}%".format(int(conf_score * 100)))
