
"""
Cat vs Dog prediction/testing script.
Run with your venv active: python predict.py
"""
 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
 
# ---------------------------------------------------------
# 1. Load the trained model
# ---------------------------------------------------------
MODEL_PATH = 'model.h5'  # <-- change this if your saved model has a different name/path
 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'. "
        f"Make sure you saved it after training with model.save('{MODEL_PATH}')."
    )
 
model = load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")
 
# ---------------------------------------------------------
# 2. Test on a single image
# ---------------------------------------------------------
image_path = 'premium_photo-1666777247416-ee7a95235559.avif'
 
if os.path.exists(image_path):
    test_img = cv2.imread(image_path)
 
    if test_img is None:
        # cv2.imread returns None (not an exception) if it can't decode the file.
        # AVIF in particular isn't supported by all OpenCV builds.
        raise ValueError(
            f"cv2.imread couldn't decode '{image_path}'. "
            f"If this is an .avif file, your OpenCV build may not support it — "
            f"try converting it to .jpg/.png first, or install 'opencv-python' "
            f"with AVIF support / use Pillow instead to load it."
        )
 
    # Convert BGR (OpenCV default) to RGB for correct matplotlib plotting
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    plt.imshow(test_img_rgb)
    plt.axis('off')
    plt.show()
 
    # Preprocess the image to fit model expectations
    test_img_resized = cv2.resize(test_img, (256, 256))
    test_input = test_img_resized.reshape((1, 256, 256, 3))
 
    # Scale image pixels if the training pipeline scales them
    test_input = test_input / 255.0
 
    # Prediction
    prediction = model.predict(test_input)
    print(f"Raw Prediction Value: {prediction[0][0]}")
 
    if prediction[0][0] < 0.5:
        print("Prediction: It's a CAT!")
    else:
        print("Prediction: It's a DOG!")
else:
    print(f"Image not found at {image_path}. Please check your path.")
 
