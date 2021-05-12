# Job-Change-of-Data-Scientist
## Description
A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the                          company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company after training or                          looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of                          candidates.

Email :- chiragagrawal196@gmail.com <br>
LinkedIn :- www.linkedin.com/in/chirag-agrawal-0bb939182/ <br>
Medium :- https://chiragagrawaldatascientist.medium.com/ <br>

## Table of Contents

1. [ Files Description ](#File_Description)
   - 1.1. [ Data ](#Data)
   - 1.2. [ Jupyter Notebooks ](#Jupyter_Notebooks)
   - 1.3. [ Other Important Files ](#Other_Important_Files)
2. [ Technologies used ](#Technologies_Used)
3. [ Structure of Notebook ](#Structure_of_Notebook)

## Files Description
 <a name="File_Description"></a>
 
 ### [Data](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/tree/master/Data)
 <a name="Data"></a>
- [aug_train.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/aug_train.csv) :- The initial training data I downloaded from Kaggle.com. <br>
- [aug_test_data](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/aug_test.csv) :- This is initial test data I downloaded from Kaggle.com. <br>
- [X.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/X.csv) :- After splitting training data into X and y, I converted X part into csv format. This includes all the independent features(columns).<br>
- [y.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/y.csv) :- After splitting training data into X and y, I converted y part into csv format. This includes dependent feature(target column).<br>
- [X_train.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/X_train.csv) :-  X is divided into two parts one part is X_train which includes the 75% of X data for training the model.<br>
- [X_test.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/X_test.csv) :-  X is divided into two parts another part is X_test which includes the 25% of X data for validating the model.<br>
- [y_train.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/y_train.csv) :-  y is divided into two parts one part is y_train which includes the 75% of y data for training the model.<br>
- [y_test.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/y_test.csv) :-  y is divided into two parts another part is y_test which includes the 25% of y data for validating the model.<br>
 
 - [X_svm_smote.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/X_svm_smote.csv) :- This is similar to X.csv but after applying SVMSmote <br>
 - [y_svm_smote.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/y_svm_smote.csv) :- This is similar to y.csv but after applying SVMSmote <br>
 - [X_train_svm_smote.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/X_train_svm_smote.csv) :- This is similar to X_train.csv but after applying SVMSmote means it is an oversampled data <br>
 - [X_test_svm_smote.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/X_test_svm_smote.csv) :- This is similar to X_test.csv but after applying SVMSmote means it is an oversampled data <br>
 - [y_train_svm_smote.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/y_train_svm_smote.csv) :- This is similar to y_train.csv but after applying SVMSmote means it is an oversampled data <br>
 - [y_test_svm_smote.csv](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/y_test_svm_smote.csv) :- This is similar to y_test.csv but after applying SVMSmote means it is an oversampled data <br>
 - [cleaned_train_data](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Data/cleaned_train_data.csv) :- This is a cleaned train data obtain after MICE Imputation. <br>
 
 ### [Jupyter Notebooks](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/tree/master/Jupyter%20notebooks)
 <a name="Jupyter_Notebooks"></a>
 
 - [Solving Class Imbalance via SVMSmote and Model Implementation.ipynb](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Jupyter%20notebooks/Solving_Class_Imbalance_via_SVMSmote_and_Model_Implementation.ipynb) :-  This ipynb file solves the class imbalance Problem using SVMSmote which is a variant of Smote but with a working of SVM. After applying SVMSmote, I implemented the EnsembleClassifier Machine Learning Model from imblearn library. This  Classification model too can also perform oversampling/undersampling. 
 - [Cleaning_and_MICE_Imputation.ipynb](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Jupyter%20notebooks/Cleaning_and_MICE_Imputation.ipynb) :- Here I have cleaned the data, cleaned some human error, Label Encoded the data and performed Missing Value Imputation which is a multiple column imputation nd it is generally better than single column imputation.<br>
- [Models and Tuning.ipynb](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Jupyter%20notebooks/Models_and_Tuning.ipynb) :- I have tuned different models and selected the final model with highest f1 score.<br>
- [Complete code with final model.ipynb](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/Jupyter%20notebooks/Complete_code_with_final_model.ipynb) :- This is the complete combined code in jupyter notebook with selected final model.<br>

### [Other Important Files](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist)
<a name="Other_Important_Files"></a>

- [EasyEnsembleClassifier_with_LGBMClassifier_as_base_estimator.pickle](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/EasyEnsembleClassifier_with_LGBMClassifier_as_base_estimator.pickle) :- This is a final model pickle file.<br>
- [HRmodel.py](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/HRmodel.py) :- This is same as "file Complete Code with final model.ipynb" but I have used FastAPI for the deployement using Pycharm, so I just created a .py file with same code.<br>
- [HRapp.py](https://github.com/ChiragAgrawalDataScientist/Job-Change-of-Data-Scientist/blob/master/HRapp.py) :- This is an app file used for deployement created using FastAPI library.<br>

 
## Technologies Used 
 <a name="Technologies_Used"></a>
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

 
