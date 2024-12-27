# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# def predict_quality(image_path, model, categories):
#     img = load_img(image_path, target_size=(128, 128))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     predicted_index = np.argmax(prediction)
#     if predicted_index >= len(categories):
#         return "Invalid category index"
#     return categories[predicted_index]

# if __name__ == "__main__":
#     model = load_model('models/fruit_quality_model_ann.keras')
#     categories = ['Export Quality', 'Domestic Quality', 'Rejected']
#     image_path = '/Users/dipaktomar/Documents/rotten_tomato.jpeg' 
#     result = predict_quality(image_path, model, categories)
#     print(f"Predicted Quality: {result}")
#     print(model.summary())


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('models/fruit_quality_model_ann.keras')
categories = ['Export Quality', 'Domestic Quality', 'Rejected']

def predict_quality(image, model, categories):
    """Predict the quality of the uploaded image."""
    img = image.resize((128, 128))  # Resize to model input size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    if predicted_index >= len(categories):
        return "Invalid category index"
    return categories[predicted_index]

# Streamlit UI
st.title("Fruit and Vegetable Quality Classifier")

st.write("Upload an image of a fruit or vegetable to predict its quality category:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = predict_quality(image, model, categories)
    st.write(f"Prediction: **{prediction}**")

