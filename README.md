# Brain-Tumor-Classification-using-CNN

🧠 Brain Tumor MRI Classification Using Deep Learning
This project provides a Streamlit-based web application for classifying brain tumor types from MRI scans. The app uses a deep learning model (e.g., CNN) trained to classify brain tumor images into one of the following categories:
1. No Tumor
2. Glioma
3. Meningioma
4. Pituitary Tumor

Features
Upload and classify MRI images of brain scans.
Select different pre-trained models for classification.
Visualize the regions contributing to the classification decision using Grad-CAM heatmap.
Display prediction probabilities and confidence scores.

📦 Installation
Follow these steps to set up the app locally:
Prerequisites

Ensure you have the following installed:
Python 3.7+
pip (Python package manager)

Step 1: Clone the Repository
Clone this repository to your local machine:
bash
Copy
Edit
git clone https://github.com/your-username/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier

Step 2: Install Dependencies
Create a virtual environment and install the required dependencies:
pip install -r requirements.txt

The requirements.txt should include the following dependencies:
streamlit
tensorflow
opencv-python
matplotlib
numpy

Step 3: Add Your Model and Images
Place your trained model (e.g., best_model.h5 or model_1.keras) in the models/ directory.
You can upload your own images directly through the Streamlit UI.

🚀 Running the App
Open a terminal/command prompt and navigate to the project folder.
Run the app with the following command:
streamlit run app.py
This will start the app locally, and you can access it by visiting http://localhost:8501 in your browser.

🧑‍💻 Using the App
Upload MRI Image:
Click on the "Upload MRI Image" button and select an MRI image file (JPG, JPEG, or PNG).
The app will display the uploaded MRI scan.

Select Model:
From the "Select a Model" dropdown, choose the model you want to use for prediction (if there are multiple models available).

Generate Prediction:
Click on the "Generate Predictions" button to classify the image.
The app will show the predicted tumor type and confidence score.
A bar chart will display the probability for each class (tumor type).

Grad-CAM Visualization:
The app will generate a Grad-CAM heatmap showing which regions of the MRI image the model used to make its decision.
This heatmap is overlaid on the original MRI image for better visualization.

⚙️ How It Works
The app utilizes a Convolutional Neural Network (CNN) model for classifying brain tumor types. It also implements Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations for the model’s predictions. The key steps are:

Preprocessing: The input image is preprocessed (resized and normalized) for prediction.
Model Inference: The model processes the image and provides the predicted tumor type and its corresponding confidence.
Grad-CAM: Grad-CAM highlights the regions in the MRI scan that contributed the most to the prediction by generating a heatmap.

📊 Results
The app provides the following outputs after prediction:
Predicted Class: The model's classification of the MRI scan (e.g., Glioma, Meningioma, Pituitary, or No Tumor).
Confidence: The model's confidence in its prediction, displayed as a percentage.
Prediction Bar Chart: A graph showing the probability of each class (tumor type).
Grad-CAM Visualization: A heatmap that overlays on the original MRI scan to explain the model's decision.

🛠️ Future Enhancements
Multiple Model Support: Allow users to switch between different pre-trained models.
Enhanced UI/UX: Improve user interface and experience for better interaction with the app.
Model Retraining: Implement a feature to retrain the model directly through the app using new MRI scans.
