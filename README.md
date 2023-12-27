# Bangalore House Price Prediction

This project aims to predict house prices in Bangalore using machine learning. The dataset used for this project is available on Kaggle [here](https://www.kaggle.com/datasets/bandhansingh/banglore-house-price-data).

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Outlier Removal](#outlier-removal)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)
- [Files and Directories](#files-and-directories)
- [Dependencies](#dependencies)

## Dataset
The dataset contains information about house prices in Bangalore, including features such as location, size, total square feet, number of bathrooms, and price.

## Data Preprocessing
The data preprocessing steps include handling missing values, converting size and total square feet to numeric values, and creating new features.

## Feature Engineering
Feature engineering involves creating new features, such as the number of bedrooms (BHK), and handling outliers in the dataset.

## Outlier Removal
Outliers are removed based on business logic and statistical methods to improve model accuracy.

## Model Training
Linear Regression is used to train the model for predicting house prices. Other models such as Lasso and Decision Tree are also evaluated.

## Model Evaluation
The models are evaluated using various performance metrics, and the best model is selected using GridSearchCV.

## Model Deployment
The trained model is saved using pickle, and column information is saved in a JSON file for deployment.

## Files and Directories
- `banglore_real_estate_data.csv`: Raw dataset.
- `house_price_prediction.ipynb`: Jupyter Notebook containing the code.
- `banglore_home_prices_model.h5`: Saved model.
- `columns.json`: JSON file containing column information.

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn

# Bangalore Home Price Prediction API

This Flask application serves predictions for house prices in Bangalore using a trained machine learning model. The model predicts home prices based on input parameters such as total square footage, location, number of bedrooms (BHK), and number of bathrooms.

## Getting Started

To get started, follow the instructions below to run the Flask server on your local machine.

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Flask
- NumPy
- scikit-learn

Install dependencies using the following command:

```bash
pip install -r requirements.txt
