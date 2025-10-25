"""
Streamlit application for Cats vs Dogs Image Classification
This app uses a trained CNN model to classify uploaded images as either cats or dogs.
"""

import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import os

@st.cache_resource
def load_model_and_preprocessing():
    """
    Load the trained model and preprocessing information.
    Uses Streamlit's cache_resource decorator for efficiency.
    """
    try:
        # Load the trained Keras model
        model_path = 'best_model_cats_vs_dogs.keras'
        model = tf.keras.models.load_model(model_path)
        
        # Load preprocessing information
        preprocessing_info = joblib.load('preprocessing_info.joblib')
        
        return model, preprocessing_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, preprocessing_info):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image: PIL Image object
        preprocessing_info: Dictionary containing preprocessing parameters
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to model's expected dimensions
    image = image.resize((preprocessing_info['image_width'], preprocessing_info['image_height']))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply rescaling (normalization)
    img_array = img_array * preprocessing_info['rescale']
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, preprocessing_info, processed_image):
    """
    Make prediction on the processed image.
    
    Args:
        model: Loaded Keras model
        preprocessing_info: Dictionary containing class names and other info
        processed_image: Preprocessed image array
    
    Returns:
        Tuple of (predicted_class_name, confidence_score)
    """
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    
    # Get predicted class index (0 for cats, 1 for dogs based on threshold 0.5)
    predicted_class_index = int(prediction[0][0] > 0.5)
    
    # Get class name
    predicted_class_name = preprocessing_info['class_names'][predicted_class_index]
    
    # Calculate confidence score and convert to Python float
    confidence = float(prediction[0][0]) if predicted_class_index == 1 else float(1 - prediction[0][0])
    
    return predicted_class_name, confidence

def main():
    """
    Main Streamlit application function.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Cats vs Dogs Classifier",
        page_icon="🐾",
        layout="centered"
    )
    
    # Application title and description
    st.title("🐾 Cats vs Dogs Image Classifier")
    st.markdown("""
    Upload an image of a cat or dog, and this AI model will predict which animal it is!
    
    **How it works:**
    1. Upload an image using the file uploader below
    2. Click the "Predict" button
    3. See the prediction result and confidence score
    """)
    
    # Load model and preprocessing info
    model, preprocessing_info = load_model_and_preprocessing()
    
    if model is None or preprocessing_info is None:
        st.error("Failed to load the model. Please check if the model files are available.")
        return
    
    # Display model information
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Classes:** {', '.join(preprocessing_info['class_names'])}")
        st.write(f"**Input Image Size:** {preprocessing_info['image_width']}x{preprocessing_info['image_height']} pixels")
        st.write(f"**Model Type:** Convolutional Neural Network (CNN)")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        help="Upload an image of a cat or dog (supported formats: JPG, JPEG, PNG, BMP, GIF)"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Prediction")
            
            # Predict button
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess the image
                        processed_image = preprocess_image(image, preprocessing_info)
                        
                        # Make prediction
                        predicted_class, confidence = predict_image(model, preprocessing_info, processed_image)
                        
                        # Display results
                        st.success("Prediction Complete!")
                        
                        # Create a nice result display
                        if predicted_class.lower() == 'cats':
                            emoji = "🐱"
                            color = "blue"
                        else:
                            emoji = "🐶"
                            color = "orange"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                            <h2 style="color: {color};">{emoji} {predicted_class.title()}</h2>
                            <h3>Confidence: {confidence:.2%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(confidence)
                        
                        # Additional information
                        if confidence > 0.8:
                            st.info("🎯 High confidence prediction!")
                        elif confidence > 0.6:
                            st.warning("⚠️ Moderate confidence prediction.")
                        else:
                            st.warning("🤔 Low confidence prediction. The image might be unclear or contain both animals.")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    # Add some information in the sidebar
    with st.sidebar:
        st.header("About This App")
        st.write("""
        This application uses a Convolutional Neural Network (CNN) trained to classify images of cats and dogs.
        
        **Features:**
        - Real-time image classification
        - Confidence score for predictions
        - Support for multiple image formats
        - User-friendly interface
        
        **Tips for better results:**
        - Use clear, high-quality images
        - Ensure the animal is clearly visible
        - Avoid images with multiple animals
        """)
        
        st.header("Model Details")
        if preprocessing_info:
            st.write(f"**Image Input Size:** {preprocessing_info['image_width']}×{preprocessing_info['image_height']}")
            st.write(f"**Classes:** {len(preprocessing_info['class_names'])}")
            st.write(f"**Architecture:** CNN with Keras/TensorFlow")

if __name__ == "__main__":
    main()