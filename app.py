import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st

# CONSTANTS
CLASS_NAMES = ['notumor', 'glioma', 'meningioma', 'pituitary']
TARGET_SIZE = (128, 128)
MODEL_PATH = "models"

# Load model
@st.cache_resource
def load_selected_model(model_file):
    return tf.keras.models.load_model(os.path.join(MODEL_PATH, model_file))

# Preprocess image for prediction using OpenCV
def preprocess_image(img):
    # Convert the image to RGB and resize using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB
    img = cv2.resize(img, TARGET_SIZE)  # Resize using OpenCV
    img = img.astype("float32") / 255.0  # Normalize the image
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    return img_array

# Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(original_image, heatmap, alpha=0.4):
    # Resize the heatmap to match the image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Scale the heatmap to [0, 255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap

    # Superimpose the heatmap on the image
    superimposed = heatmap * alpha + original_image
    superimposed = np.uint8(superimposed)  # Convert to uint8 for display
    return superimposed

# App
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload an MRI scan and select a model to classify it as one of the 4 tumor types.")

# Select model
model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(('.h5', '.keras'))]
selected_model_file = st.selectbox("Select a Model", model_files)

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read image using OpenCV
    image_bytes = uploaded_file.read()
    img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode the image to BGR format
    
    # Display original image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    # Preprocess image for prediction
    img_array = preprocess_image(image)

    # Load model
    model = load_selected_model(selected_model_file)

    # Predict
    preds = model.predict(img_array)[0]
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = 100 * np.max(preds)

    st.markdown(f"### 🔍 Prediction: `{pred_class}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")

    # Plot probabilities
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, preds, color='skyblue')
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    # Grad-CAM
    st.markdown("### 🧠 Grad-CAM Visualization")
    try:
        # Try common conv layer names
        conv_layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=conv_layer_names[-1])
        
        # Grad-CAM visualization
        cam_image = display_gradcam(image, heatmap)
        st.image(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB), caption='Grad-CAM', use_container_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
