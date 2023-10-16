# EQ_Damage_Prediction_Project
Analysis and Prediction of severity of damage to residential buildings caused by Earthquake using Machine learning model

In the real world, Catastrophe risk modeling combines historical disaster information with current demographics, building, and financial data to determine the potential financial impact of catastrophe in various geographic locations. However, the process of developing sophisticated catastrophe models is complex and draws on expertise from a broad range of technical and financial disciplines. In this project, we will limit ourselves to one peril Earthquake and just use one type of occupancy ‘residential' for educational purposes and realize the power of machine learning in predictive analytics.

We will preprocess and explore historical earthquake event data and build a machine-learning model to identify the severity of damage to buildings caused by an earthquake. Additionally, we will merge household conditions data to capture the socio-economic and demographics of affected areas and see if they improve scores in our classification task. This is useful for Property Insurers to assign different rates based on the expected severity of damage to client property that is identified by our machine learning model.

## Problem Statement

Given the geo-location, structure details of the building, and household socio-economic demographics information predict the severity of damage to buildings caused by Earthquake

## Business Constraint

- Interpretability is important to some extent
- No strict low latency concerns
- High accuracy. Errors affect the pricing of premiums and can cost writing business
- Predicting the probability of a point belonging to each class is not necessary

## Data Overview

(Source: Nepal Earthquake 2015 open data portal (link: https://reliefweb.int/report/nepal/open-data-portal-2015-earthquake-launched-national-planning-commission))
The data mainly contains information on the structure of buildings like the construction material used for the foundation/roof, height of the building, plinth area, surface, age, their legal ownership, geo-location and household conditions, income level, etc. to name a few for the eleven severely affected districts of Nepal. Features are self-explanatory by their names. We will explore each feature in the EDA notebook.


## Performance Metric

- Both precision and recall are important so the Micro-F1 score is a good choice as data is imbalanced and we care about the overall accuracy
- Confusion matrix, recall matrix, and precision matrix to see how our model is performing on train and test data for each class
- Classification report to see how our model is performing in each class

## Approach

1. Fetch the building structure data and household data from the website and preprocess them to get clean and consistent data
2. Split the preprocessed data into train and test set
3. Explore the training data to understand it and identify any interesting relations
5. Remove outlier data points if any from the data
6. Apply feature engineering techniques like Boxcox transformation on the numerical features that are skewed, entity embedding for the categorical features using simple Neural Network
7. Balance the data. Try oversampling (SMOTE), undersampling (TomekLinks), and Combinedsampling (SMOTETomek)
8. Scale the data (Standardization or Min-Max scaling)
9. Try different models. Linear models like logistic regression, SVM, non-linear models like Decision Trees, and ensemble models like Random Forest, Gradient Boosting, etc
10. Hyperparameter tune each model using either grid search, random search, or TuneSearch (Bayesian optimization) to get the optimum hyperparameters
11. Use Micro_F1 score as primary metric and use confusion matrix and classification report as secondary metrics
12. Identify the most useful features using feature importance techniques
13. Rerun the data with the best model and with top features for prediction and scoring

## Key Techniques

- BoxCox transformation on some of the highly skewed numerical features 
- Entity Embedding for categorical features
- Used SMOTE, TomekLink, and SMOTETomek data balancing techniques. Chose best one on the performance of entity embedding model
- High efficient gradient boosting algorithm - LightGBM is used

##  Files

There are 5 jupyter notebook files
1. Data Cleaning                    - 1_Nepal Earthquake 2015_Data Cleaning.ipynb
2. Exploratory Data Analysis        - 2_Nepal Earthquake 2015_EDA.ipynb
3. Data Preprocessing               - 3_Nepal Earthquake 2015_Preprocessing.ipynb
4. Data Modeling                    - 4_Nepal Earthquake 2015_Modeling.ipynb
5. Predict Function for deployment  - 5_Nepal Earthquake 2015_Predict function.ipynb

## Inference

1. The performance of linear models is lower than that of our embedding neural network model as expected
2. Random Forest is performing on par with that of our embedding neural network model
3. Lightgbm is outperforming all other models

   ![image](https://github.com/KiranMahendrakar93/EQ_Damage_Prediction_Project/assets/88178398/1a2186df-ea41-4909-b714-de41b25fbebb)

4. Below is the confusion matrix and classification report of LightGBM model

   ![image](https://github.com/KiranMahendrakar93/EQ_Damage_Prediction_Project/assets/88178398/f797704a-b712-4c5c-8f65-3aa609be157c)

   ![image](https://github.com/KiranMahendrakar93/EQ_Damage_Prediction_Project/assets/88178398/ea8c695b-34ed-4e00-9413-48d7ba281639)

5. we can see that LightGBM model is able to generalize well on majority class ‘Severe’ followed by ‘mild’ and overfitting on ‘moderate’ class
6. Feature Importance

   ![image](https://github.com/KiranMahendrakar93/EQ_Damage_Prediction_Project/assets/88178398/ebfb5610-d148-4f6d-97d4-699980dcd0e4)

  ![image](https://github.com/KiranMahendrakar93/EQ_Damage_Prediction_Project/assets/88178398/0b9dd6c6-2f48-4912-9714-51ee5d1d698e)

7. Top 5 features are the age_of_building, the height_of_building, plinth_area, size_of_household, and age_of_household_head
8. Building structure details like geo-location, age, area, number of floors, construction, etc play a vital role in determining the severity of damage
9. Interestingly we can see that socio-economic features like size of household, age of household head, education level, type of toilet, source of water, etc are also playing important roles in determining the severity of damage
10. Embedded features obtained using neural networks also capture most of the variance in the data
11. The performance of the model decreases if we reduce the number of dimensions as most features are required to explain the majority of the variance
 
## Blog

Link - 
