# Cyclone Intensity CV Estimator 🌀

This project is an automated AI system that takes raw satellite Infrared imagery (INSAT-3D/TCIR) and instantly predicts the maximum sustained wind speed (intensity) of a tropical cyclone. 

This repository was built for the AI for Social Good Hackathon, addressing **SDG 13 (Climate Action)** and **SDG 11 (Sustainable Cities)**.

## Features
*   **Deep Learning Inference:** A custom Convolutional Neural Network (CNN) built in TensorFlow/Keras tailored for regression tasks.
*   **Early Warning Dashboard:** An interactive web frontend built with Streamlit for meteorologists to load images and see instant predictions.
*   **Interactive Forecasts:** Automatically charts an intensity trajectory curve over 24 hours.

## Local Setup
1. Clone this repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

## Note on Dataset
The `TCIR-ALL_2017.h5` dataset used to train the model is 2.9GB and is excluded from this repository via `.gitignore`. 
