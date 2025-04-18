# Brain-Tumor-Classification-using-CNN

## Brain Tumor MRI Classification Using Deep Learning
This project provides a Streamlit-based web application for classifying brain tumor types from MRI scans. The app uses a deep learning model (e.g., CNN) trained to classify brain tumor images into one of the following categories:
1. No Tumor
2. Glioma
3. Meningioma
4. Pituitary

## Features
1. Upload and classify MRI images of brain scans.
2. Select different pre-trained models for classification.
3. Display prediction probabilities and confidence scores.

## Installation
Follow these steps to set up the app locally:

## Step 1: Install Dependencies
Create a virtual environment and install the required dependencies:
pip install -r requirements.txt

The requirements.txt should include the following dependencies:
1. opencv-python
2. numpy
3. pandas
4. matplotlib
5. streamlit
6. tensorflow

## Step 2: Add Your Model and Images
1. Place your trained model (e.g., best_model.h5 or model_1.keras) in the models/ directory.
2. You can upload your own images directly through the Streamlit UI.

## Running the App
1. Open a terminal/command prompt and navigate to the project folder.
2. Run the app with the following command:
3. streamlit run app.py
4. This will start the app locally, and you can access it by visiting http://localhost:8501 in your browser.

## Using the App
1. Upload MRI Image:
2. Click on the "Upload MRI Image" button and select an MRI image file (JPG, JPEG, or PNG).
3. The app will display the uploaded MRI scan.

## Select Model:
From the "Select a Model" dropdown, choose the model you want to use for prediction (if there are multiple models available).

## Generate Prediction:
1. Click on the "Generate Predictions" button to classify the image.
2. The app will show the predicted tumor type and confidence score.
3. A bar chart will display the probability for each class (tumor type).

## How It Works
The app utilizes a Convolutional Neural Network (CNN) model for classifying brain tumor types. It also implements Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations for the model’s predictions. The key steps are:
1. Preprocessing: The input image is preprocessed (resized and normalized) for prediction.
2. Model Inference: The model processes the image and provides the predicted tumor type and its corresponding confidence.

## Results
The app provides the following outputs after prediction:
1. Predicted Class: The model's classification of the MRI scan (e.g., Glioma, Meningioma, Pituitary, or No Tumor).
2. Confidence: The model's confidence in its prediction, displayed as a percentage.
3. Prediction Bar Chart: A graph showing the probability of each class (tumor type).

## Future Enhancements
1. Multiple Model Support: Allow users to switch between different pre-trained models.
2. Enhanced UI/UX: Improve user interface and experience for better interaction with the app.
3. Model Retraining: Implement a feature to retrain the model directly through the app using new MRI scans.
