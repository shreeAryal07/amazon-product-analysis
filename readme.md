# Amazon Product Data Analysis

My first end-to-end beginner-level data analysis project using a real-world Amazon product dataset.

## Objectives
- Clean and preprocess product data.
- Perform Exploratory Data Analysis (EDA).
- Analyze brands, categories, prices, discounts, and ratings.
- Create visualizations to discover useful patterns.
- Prepare for future Machine Learning and AI-powered product analysis.

## Tools & Technologies
- Python
- Pandas
- Matplotlib
- Seaborn
- VS Code
- Git and GitHub

## Work Completed
- Inspected dataset structure and missing values.
- Converted rating counts, ratings, prices, and discount percentages into numeric formats.
- Handled missing values.
- Extracted brand names from product names.
- Created main category, subcategory, and short product name columns.
- Explored top brands, categories, and product ratings.
- Created a multi-chart visualization dashboard.
- Saved cleaned data as a CSV file.

## Future Plans
- Build a suitable Machine Learning prediction model.
- Develop an interactive product filtering and sorting tool.
- Explore an AI-powered product assistant that recommends products based on user requirements.
- Improve the dashboard and deploy the application.

## Learning Journey
This project is part of my journey toward becoming a Data Scientist. I am learning through hands-on projects and continuously improving my data analysis and Machine Learning skills.

**Status:** Work in progress 🚀

*Note: Product popularity based on rating counts does not necessarily represent actual sales.*


## Machine Learning: Rating Prediction

A Machine Learning model was developed to predict Amazon product ratings using product prices, discount percentage, rating counts, brand, and product category.

### Models Evaluated
- Linear Regression
- Random Forest Regressor
- Extra Trees Regressor
- Baseline Model

### Model Evaluation
Models were compared using:
- **Mean Absolute Error (MAE)**
- **R² Score**

The best-performing model was selected based on MAE.

### Project Files
- `model_training.py` — Model training and evaluation
- `amazon_cleaned.csv` — Cleaned dataset
- `rating_predictions.csv` — Actual and predicted ratings
- `actual_vs_predicted.png` — Prediction visualization

### Technologies Used
Python, Pandas, Scikit-learn, Matplotlib, Joblib

### Future Improvements
- Improve prediction performance through feature engineering and hyperparameter tuning.
- Build an interactive application for product rating prediction.
