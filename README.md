# :credit_card: SC1015_DSAI_MiniProject_CreditCardFraud :credit_card:

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on Credit card Fraud from 
[Credit Card Transactions Fraud Detector Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv). 

## Project Aim and Motivation
The aim of our project is to leverage machine learning algorithms to accurately detect fraudulent credit card transactions to minimize 
the potential costs incurred from fraud transactions. 

## Contributors
- [@adrian-ang97](https://github.com/adrian-ang97) – XGBoost, Slides
- [@AgnesT2002](https://github.com/AgnesT2002) – Data Balancing, K-Nearest Neighbors
- [@Jabezng2](https://github.com/Jabezng2) – Exploratory Data Analysis, Feature Engineering, Gaussian Naïve Bayes,     Random Forest Classifier

## Project Overview
### 1. Problem Definition ###
- Are we able to predict if a credit card transaction is fraudulent?
- What are some of the possible factors that can help in detection?
- Which model would be the best to predict it?

### 2. Data Extraction & Exploratory Data Analysis ###
- Extracted 1% from raw data due to overwhelming volume of observations
- No **NULL** or duplicate values
- Dropped unnecessary variables
- Explored variables like gender and category and the general relation to *'is_fraud'*
- Analyzed the class imbalance issue with *'is_fraud'*
- Utilized heatmap, barplot, histogram and violin plots for data visualization
- Statistical distribution of the variable amount

### 3. Feature Engineering ###
- We needed to generate new features to improve correlation and obtain a better model
- Extracted day of week, month number, year, hour, minute and second from *'trans_date_trans_time'* variable
- Encoding time periods. Transactions taking place between **2100H - 0500H** encoded as 1. Else it is encoded as 0.
- Frequency of transactions in the last day, week and month which is then aggregated with respect to *'cc_num'* and *'merchant'*
- Time since last transaction (Recency) was also obtained and aggregated with respect to *'cc_num'* and *'merchant'* too
- Age can be obtained from date of birth and datetime of transaction
- One-hot encoding was performed on *'gender'* and '*category*' due to their small number of unique values
- Label encoding performed on *'job'* and *'merchant'* due to the large number of unique values
- Distance can be calculated from the latitude and longitude values of merchant and individual using *haversine* function

### 4. Resolving Class Imbalance ###
- Looked into 3 techniques namely under sampling, oversampling and SMOTE
- Considered the disadvantages of under and over sampling methods and decided that SMOTE would be the best way forward

### 5. Model Building ###
- Build and tested 4 machine learning algorithms, Gaussian Naïve Bayes, Random Forest Classifier, K-Nearest Neighbors and XGBoost
- We used classification accuracy, TPR (Sensitivity), FPR (Specificity), F1 score and AUROC evaluation as metrices for deciding the best model
- XGBoost had the highest value for classification accuracy (1.0 for Train, 0.99910 for Test), TPR (1.0 for both Train and Test), F1 score (1.0 for Train, 0.99912 for Test) and AUC value of 1.0. It also had the lowest FPR (0.0 for Train, 0.0018 for Test)

### 6. Conclusion ###
- Feature engineering improved correlation and overall model accuracy
- Using SMOTE to balance our data improved model performance, especially on the minority class
- Fraudulent transactions are likely to be done at ungodly hours of the day with a higher amount of money involved
- The higher the frequency of transactions made from a card raises justified suspicion
- The more recent the transaction was made in relation to the last transaction made also raises the high possibility of fraud
- XGBoost did exceptionally well in predicting fraudulent transactions

### 7. New Insights Learned ###
- Feature engineering techniques such as feature aggregation and scaling (Min-max, Standard, Robust) to tackle poor correlation between variables
- Handling imbalanced datasets using various balancing techniques (Under-sampling, Oversampling, SMOTE)
- Gaussian Naïve Bayes, Random Forest Classifier, K-Nearest Neighbors, XGBoost
- Concepts on ROC & AUC, and F1 Score

## References ##
- https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
- https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
- https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/#:~:text=The%20k%2DNearest%20Neighbors%20algorithm%20or%20KNN%20for%20short%20is,a%20summarized%20prediction%20is%20made
- https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/
- https://www.ritchieng.com/machine-learning-evaluate-classification-model/
- https://www.kaggle.com/code/paymanfara/credit-card-fraud-detection-supervised-learning 
