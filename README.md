# Job-Change-of-Data-Scientist
## Description
A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the                          company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company after training or                          looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of                          candidates.

Email :- chiragagrawal196@gmail.com <br>
LinkedIn :- www.linkedin.com/in/chirag-agrawal-0bb939182/ <br>
Medium :- https://chiragagrawaldatascientist.medium.com/ <br>

## Table of Contents
<details>
<summary>Show/Hide</summary>
<br>

1. [ Files Description ](#File_Description)
2. [ Technologies used ](#Technologies_Used)
3. [ Structure of Notebook ](#Structure_of_Notebook)
</details>

## Files Description
<details>
 <a name="File_Description"></a>
 <summary>Show/Hide</summary>
 <br>
 ### Data
 <details>
 <a name="Data"></a>
  
- <strong> aug_train.csv </strong> :-  The initial training data I downloaded from Kaggle.com. <br>
- <strong> Cleaning_and_MICE_Imputation.csv </strong> :- Here I have cleaned the data, cleaned some human error, Label Encoded the data and performed Missing Value Imputation which is a multiple                                          column imputation nd it is generally better than single column imputation.<br>
- <strong> cleaned_train_data.csv </strong> :-  Cleaned aug_train data after MICE Imputation used for missing value imputation<br>

- <strong> X.csv </strong> :-        After splitting training data into X and y, I converted X part into csv format. This includes all the independent features(columns).<br>
- <strong> y.csv </strong> :-        After splitting training data into X and y, I converted y part into csv format. This includes dependent feature(target column).<br>
- <strong> X_train.csv </strong> :-  X is divided into two parts one part is X_train which includes the 75% of X data for training the model.<br>
- <strong> X_test.csv </strong>  :-  X is divided into two parts another part is X_test which includes the 25% of X data for validating the model.<br>
- <strong> y_train.csv </strong> :-  y is divided into two parts one part is y_train which includes the 75% of y data for training the model.<br>
- <strong> y_test.csv </strong>  :-  y is divided into two parts another part is y_test which includes the 25% of y data for validating the model.<br>

 * Model is fitted on train data i.e. X_train and X_test, using this train data, model tries to find the best score depending on the model and the hyperparameters of the model 
 * Later Model performance is checked on the validation set i.e. X_test and y_test and compared with the result of Training data (data that was fitted in model) 
 
 - <strong> Solving Class Imbalance via SVMSmote and Model Implementation.ipynb </strong> :- This ipynb file solves the class imbalance Problem using SVMSmote which is a variant                                                                                              of Smote but with a working of SVM. After applying SVMSmote, I implemented the                                                                                                    EasyEnsembleClassifier Machine Learning Model from imblearn library. This                                                                                                        Classification model too can also perform oversampling/undersampling. 
 - <strong> X_svm_smote.csv </strong> :- This is similar to X.csv but after applying SVMSmote <br>
 - <strong> y_svm_smote.csv </strong> :- This is similar to y.csv but after applying SVMSmote <br>
 - <strong> X_train_svm_smote.csv </strong> :- This is similar to X_train.csv but after applying SVMSmote means it is an oversampled data <br>
 - <strong> X_test_svm_smote.csv </strong> :- This is similar to X_test.csv but after applying SVMSmote means it is an oversampled data <br>
 - <strong> y_train_svm_smote.csv </strong> :- This is similar to y_train.csv but after applying SVMSmote means it is an oversampled data <br>
 - <strong> y_test_svm_smote.csv </strong> :- This is similar to y_test.csv but after applying SVMSmote means it is an oversampled data <br>
 - <strong> svm_smote_spplied_train_data.csv </strong> :- After applying SVMSmote and joing X_svm_smote & y_svm_smote we get a new dataframe which is saved as csv <br>
 </details>
 </details>
 
## Technologies Used
<details> 
 <a name="Technologies_Used"></a>
 <summary>Show/Hide</summary>
1. Python <br>
2. NumPy <br>
3. MatplotLib <br>
4. Seaborn <br>
5. scikit-learn <br>
6. imblearn <br>
7. eli5 (for Feature Importance of model) <br>
8. pickle <br>
</details>

## Structure of Notebook
<details>
 <a name="Structure_of_Notebook"></a>
 <summary>Show/Hide</summary>
1. Cleaning and MICE Imputation<br>
   - 1.1 Imports<br>
   - 1.2 Deleting unwanted columns<br>
   - 1.3 Cleaning Human error<br>
   - 1.4 Checking for nulls
   - 1.5 Label Encoding data and joining the preprocessed data & Training data<br>
   - 1.6 Missing Value Imputation using MICE.<br>
   - 1.7 Saving Cleaned structured data in csv format as cleaned_train_data.csv file.<br>
   - 1.8 Saving Structured Dataset as a CSV<br>
2. Solving Class Imbalance via SVMSmote and Model Implementation <br>
   - 2.1 Import Libraries <br>
   - 2.2 Importing cleaned_train_data.csv file <br>
   - 2.3 Checking for nulls <br>
   - 2.4 Checking for class imbalance <br>
   - 2.5 Checking intercorrealtion issue using Heatmap <br>
   - 2.6 Splitting into X and y and later standardizing it <br>
   - 2.7 Applying SVMSmote and again checking the class imbalance <br>
   - 2.8 Defining some functions for evaluating Machine Learning Model <br>
   - 2.9 Applying EasyEnsembleClassifier Machine Learning Model <br>
   - 2.10 Printing f1 score and Feature Importance according to the model using permutation importance of eli5 library <br>
   - 2.11 Pickling the Model <br>

 
