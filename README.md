# OLA---Ensemble-Learning
Recruiting and retaining drivers is seen by industry watchers as a tough battle for Ola. Churn among drivers is high and it’s very easy for drivers to stop working for the service on the fly or jump to Uber depending on the rates.

Problem Statement and Data Analysis
We are given with the driver team attrition data from Ola and are required to predict whether a driver will be leaving the company or not based on the provided attributes given that we are working as a data scientist with the Analytics Department of Ola.
The shape of the dataset looks like we have 19104 rows with 14 columns. There are few null values and categorical values present in the data.
Created barplots for Categorical variables and histogram for continuous distribution.
From the correlation metrics, it looks like Grade and Income are highly correlated. Also, Grade and Joining Designation are also highly correlated.
By describing the data, we get to know information about statistics of the data.
Analysed each features of the data and looks like some features are normally distributed.
As there are few null value rows in the dataset, hence need to fill the missing values.
The dataset is imbalanced and hence, we shouldn't just check for the accuracy of the model and also look for methods to work on imbalanced data.

        2. Data Preprocessing 
There are no duplicate values present in the dataset.
There are some missing values present in the dataset and hence used the KNN Imputer to fill the missing values.
For finding the target variable i.e. churn prediction, If NaN value is there for last working date, target would be False, else True.
Extracting years and months from date of joining and reporting date.
Grouped the data based on the driver ID column and hence, the data size gets reduced.
Prepared the data for training purpose by creating X and Y values.

        3.  Model Building
Build the model using bagging by providing the training data.
Also, built the model using Random Forest classifier and included Grid Search CV to find the best set of values to train the model.
Used the Gradient boosting classifier as well to build the model.

       4. Results Evaluation
From the ROC AUC Curve, we understand that the model is better than the dummy model and has an area under the curve of 0.90
Precision-Recall curve is mostly preferred when there is a moderate to large class imbalance and as we have data imbalance, hence preision recall values should be focused more than the ROC curve.
For model evalution, created functions such as classification_performance, kfold_cross_validation_score which provides information such as accuracy score, precision, recall and f1 core values.
From the 3 trained models, random forest classifier looks the best but all the models looks like they are overfitting the values. We can work on making such hyperparameter tunings to the model so that we can reduce overfitting to some more extent.

      5. Actionable Insights and Recommendations
There are many factors that may contributed and can be improved on are:-
a) Need more data - Current data has values, but with null values as well. Having more data worth could have helped us improve the accuracy.
b) Bad assumptions - We made the assumption that this data has a linear relationship but that might not be the case. We can do some more visualisations on the data.
c) Poor features - The features we used may not have had a high enough correlation to the values we are trying to predict.
d) Play around with the code and data to see if we can improve the results.
