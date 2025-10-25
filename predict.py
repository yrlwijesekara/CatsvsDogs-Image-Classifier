
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import os

def load_deployed_model(deployment_dir):
    model_path = os.path.join(deployment_dir, 'best_model_cats_vs_dogs.keras')
    model = tf.keras.models.load_model(model_path)
    preprocessing_info = joblib.load(os.path.join(deployment_dir, 'preprocessing_info.joblib'))
    return model, preprocessing_info

def preprocess_image(image_path, preprocessing_info):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((preprocessing_info['image_width'], preprocessing_info['image_height']))
    img_array = np.array(img)
    img_array = img_array / preprocessing_info['rescale'] # Apply rescaling
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

def predict_image(model, preprocessing_info, image_path):
    processed_image = preprocess_image(image_path, preprocessing_info)
    prediction = model.predict(processed_image)
    predicted_class_index = int(prediction[0][0] > 0.5)
    predicted_class_name = preprocessing_info['class_names'][predicted_class_index]
    confidence = prediction[0][0] if predicted_class_index == 1 else 1 - prediction[0][0]
    return predicted_class_name, confidence

if __name__ == '__main__':
    # Example Usage:
    # Assuming you have a test image at 'test_image.jpg' in the same directory as the deployment package
    # You would run this script from the deployment package directory.
    # Example: python predict.py test_image.jpg

    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    test_image_path = sys.argv[1]
    deployment_directory = '.' # Assuming script is run from deployment dir

    model, preprocessing_info = load_deployed_model(deployment_directory)
    predicted_class, confidence = predict_image(model, preprocessing_info, test_image_path)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
