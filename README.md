# 🩺 Diabetes Prediction Web App

<img width="603" height="208" alt="image" src="https://github.com/user-attachments/assets/d7fe6168-d9bc-4687-a663-6947b07c2051" />

A machine learning-powered web application that predicts the likelihood of diabetes based on patient health metrics. Built with **Streamlit** and scikit-learn.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📁 Project Structure
```bash
diabetes-prediction-app/
│
├── streamlit.py # 🖥️ Main Streamlit web application
├── train_model.py # 🤖 Model training script
├── preprocess.py # 🔧 Data preprocessing script
├── diabetes_model.joblib # 📦 Trained model (generated)
├── scaler.joblib # ⚖️ Feature scaler (generated)
├── requirements.txt # 📦 Python dependencies
└── README.md # 📖 This file
```

## 🚀 Features

- **Interactive UI** with sliders for input parameters
- **Real-time prediction** with probability scores
- **Model interpretability** with clear risk indicators
- **Responsive design** built with Streamlit
- **Scalable preprocessing** pipeline

## 📊 Input Features

The model uses 8 health metrics for prediction:

1. 🤰 Pregnancies
2. 🩸 Glucose Level
3. 💓 Blood Pressure (mm Hg)
4. 🦵 Skin Thickness (mm)
5. 💉 Insulin Level (mu U/ml)
6. ⚖️ BMI
7. 🧬 Diabetes Pedigree Function
8. 🎂 Age

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/diabetes-prediction-app.git
cd diabetes-prediction-app
```
### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
If requirements.txt is not available, install manually:
```bash
pip install streamlit scikit-learn pandas numpy joblib
```

## 📈 Model Training
1. **Preprocess the Data**
```bash
python preprocess.py
```
- Loads and cleans the dataset
- Handles missing values
- Splits data into training and testing sets

2. **Train the Model**
```bash
python train_model.py
```
- Trains a classification model (e.g., Random Forest)
- Saves model as diabetes_model.joblib
- Saves scaler as scaler.joblib

## 🌐 Running the Web App
```bash
streamlit run streamlit.py
```
The app will open in your default browser at http://localhost:8501

## 🎯 How to Use
- **Adjust Parameters:** Use the sliders to input patient health metrics
- **Click Predict:** Press the "Predict Diabetes" button
- **View Results:** See prediction (High/Low risk) with probability percentage
- **Interpret:** Read the interpretation guidelines for context

## 📈 Model Performance
The model provides:
- ✅ Binary classification (High/Low risk)
- ✅ Probability scores for better interpretation
- ✅ Scaled input features for consistent predictions

## ⚠️ Important Disclaimer
This application is for educational and demonstration purposes only.

- 🤕 Not a substitute for professional medical advice
- 🩺 Always consult healthcare professionals for medical diagnoses
- 📊 Predictions are based on statistical models, not medical expertise

## 🧪 Testing the Model
You can test with sample values:

- High Risk Profile: High glucose, high BMI, older age
- Low Risk Profile: Normal glucose, healthy BMI, younger age

## 🔧 Customization
To modify the model:

- Edit train_model.py to change algorithms or parameters
- Adjust preprocessing steps in preprocess.py
- Update feature scaling or input ranges in streamlit.py

## 🤝 Contributing
- Contributions are welcome! Please feel free to submit a Pull Request.
- Fork the repository
- Create your feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
Your Name - Humaima Anwar

## 🙏 Acknowledgments
Dataset: Diabetes Database

## Built with Streamlit
Machine learning with scikit-learn

⭐ If you find this project useful, please give it a star! ⭐

