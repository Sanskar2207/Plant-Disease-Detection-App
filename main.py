import streamlit as st
import tensorflow as tf
import numpy as np
import h5py
#Tensorflow model prediction

def model_prediction(test_image):
    model=tf.keras.models.load_model(r"C:\Users\DELL\Desktop\jupyter Notebook\New Plant disease detection\Trained_model.keras")#loading saved model
    # Image Preprocessing
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])# convert single image to batch

    prediction=model.predict(input_arr)#making prediction

    result_index=np.argmax(prediction)  

    return result_index

#Sidebarp
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])
if (app_mode=="Home"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    image_path="Plant Disease.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""

Welcome to the Crop Disease Detection System! 🌿🔍

Our goal is to help you identify crop disease issues efficiently. Upload a picture of your plant, and our system will analyze it to detect any signs of disease. Let's work together to maintain healthy crops and gardens!

### How It Works
1. **Upload Image:** Navigate to the Disease Detection page and upload a photo of your plant.
2. **Analysis:** Our system will examine the image using algorithms to identify any health issues.
3. **Results:** Receive detailed results and recommendations for further care.
                
### Why Choose Us?
- **Accuracy:** Our advanced machine learning techniques ensure precise health issue identification.
- **Easy to Use:** Our intuitive interface makes the process simple and straightforward.
- **Quick and Reliable:** Get results in moments, enabling prompt and effective interventions.

                
### Get Started
Visit the Disease Detection page in the sidebar to upload an image and discover how our Crop Disease Detection System can help you!

### About Us
Find out more about our project, team, and mission on the About page.
""")
    
#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown(""" 
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
    This dataset consists of about 7K rgb images of healthy and diseased crop leaves which is categorized into 3 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 14 test images is created later for prediction purpose.
    
    #### Content
    1. Train (5702 Images)
    2. Valid (1426 Images)
    3. Test (14 Images)
""")
    
#Detection Page
elif(app_mode=="Disease Detection"):
    st.header("Disease Detection")
    test_image=st.file_uploader("Choose an Image")

    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)

    #Predict Button
    if(st.button("Detect")):
        st.balloons()
        st.write("Our Result")
        result_index=model_prediction(test_image)
        class_name=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))