import streamlit as st
import tensorflow as tf
import numpy as np


def model_prediction(img):
    model = tf.keras.models.load_model("model/trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(img, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    # input_arr.shape
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


st.sidebar.write("Dashboard")
page_selected = st.sidebar.selectbox("Select page",["Home","Disease Detection","About us"])

if page_selected == "Home":
    st.header("Plant Leaf Disease Detection")
    img_path = "Images/home_page.png"
    st.image(img_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    ## Overview
    This web app helps users detect diseases in plant leaves by analyzing images. Simply upload a picture of the plant leaf, and the app will predict the type of disease (if any) present on the leaf.
    
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    
    ### About Us
    Learn more about the project on the **About** page.
    
    
    """)
elif page_selected == "About us":
    st.header("About Us")
    st.markdown("""
            #### About Dataset
            The dataset can be found on the kaggle.
            This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
            A new directory containing 33 test images is created later for prediction purpose.
            #### Content
            1. train (70295 images)
            2. test (33 images)
            3. validation (17572 images)
    
    """)
elif page_selected == "Disease Detection":
    st.header("Leaf Disease Detection")
    image = st.file_uploader("Choose image")
    if st.button("Show Image"):
        st.image(image,width=4,use_column_width=True)

    if st.button("Show Predict"):
        st.write("Our Prediction")
        with st.spinner("Please wait..."):

        # st.snow()
            result_index = model_prediction(image)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            result = class_name[result_index]
            st.success(f"Model is predicting it's a {result}")

    
    
    
