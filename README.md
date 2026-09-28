# 🏠 Kolkata HousePrice AI

A simple and interactive **Kolkata House Price Prediction** application built using Machine Learning and Streamlit.

The application estimates the approximate value of a residential property based on basic property details such as location, area, number of bedrooms, and property status.

---

## 👩‍💻 Developed By

**Ananya Das**

---

## 📌 Project Overview

Kolkata HousePrice AI is a practical machine learning project that demonstrates the complete workflow of a house price prediction system.

The project covers:

- Data collection
- Data preprocessing
- Missing-value handling
- Feature processing
- Machine learning model training
- Model evaluation
- Best-model selection
- Prediction testing
- Interactive Streamlit interface

---

## ✨ Key Features

- 📍 Kolkata-based property locations
- 📐 Area-based price estimation
- 🛏 Bedroom-based prediction
- 🏠 New / Resale property selection
- 💰 Estimated price displayed in Indian Rupees
- 🤖 Multiple machine learning models
- 📊 Model evaluation using MAE, RMSE and R²
- 🖥️ Interactive Streamlit interface
- ✅ Separate model testing script

---

## 🧠 Machine Learning Models

The project compares three regression models:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The model with the highest test-set **R² score** is automatically selected for the prediction interface.

---

## 🔄 Project Workflow

```text
Kolkata Housing Dataset
        ↓
Data Cleaning
        ↓
Missing Value Handling
        ↓
Feature Preprocessing
        ↓
Train / Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Model Saving
        ↓
Interactive Prediction
