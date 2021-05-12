import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score, \
    roc_auc_score, roc_curve, auc
from lightgbm.sklearn import LGBMClassifier
import six
import sys
sys.modules['sklearn.externals.six'] = six
from imblearn.over_sampling import SMOTENC, SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.ensemble import EasyEnsembleClassifier
import eli5
from eli5.sklearn import PermutationImportance
import pickle

missing_values = ["n/a", "na", "--", "NONE", "None", "none", "NA", "N/A", 'inf', '-inf', '?', 'Null', 'NULL']
train_data = pd.read_csv("E:\chirag\Datasets\Job change\Aug_train.csv", na_values=missing_values)
train_data.drop(['enrollee_id', 'city'], 1, inplace=True)

print(train_data.company_size.value_counts())
train_data['company_size'] = train_data['company_size'].replace('10/49', np.nan)
print("==============================")
print(train_data.company_size.value_counts())

to_LabelEncode = train_data[['gender', 'relevent_experience',
                             'enrolled_university', 'education_level', 'major_discipline',
                             'experience', 'company_size', 'company_type', 'last_new_job']]

le = LabelEncoder()
train_temp = to_LabelEncode.astype("str").apply(le.fit_transform)
train_Label_encode = train_temp.where(~to_LabelEncode.isna(), to_LabelEncode)

train_data.drop(['gender', 'relevent_experience', 'enrolled_university', 'education_level',
                 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job'], 1, inplace=True)

train_data = train_Label_encode.join(train_data)
print(train_data)

lr = LinearRegression()
mice_imputer = IterativeImputer(random_state=42, estimator=lr, max_iter=10, n_nearest_features=2,
                                imputation_order='roman')
cleaned_train_data = mice_imputer.fit_transform(train_data)

cleaned_train_data = pd.DataFrame(cleaned_train_data)
cleaned_train_data.columns = ['gender', 'relevent_experience', 'enrolled_university', 'education_level',
                              'major_discipline',
                              'experience', 'company_size', 'company_type', 'last_new_job', 'city_development_index',
                              'training_hours', 'target']

print(cleaned_train_data)

X = cleaned_train_data.drop('target', 1)
y = cleaned_train_data.target

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svm_smote = SVMSMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5)
X_svm_smote, y_svm_smote = svm_smote.fit_resample(X, y)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = tts(X_svm_smote, y_svm_smote, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train_svm = sc.fit_transform(X_train_svm)
X_test_svm = sc.transform(X_test_svm)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    errors = abs(y_pred - y_test)
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('Recall Score = ', recall_score(y_test, y_pred))
    print('Precision Score = ', precision_score(y_test, y_pred))
    print('F1 score = ', f1_score(y_test, y_pred))

    return evaluate


def train_auc_roc_curve(model, X_train, y_train):
    base_fpr, base_tpr, base_threshold = roc_curve(y_train, model.predict(X_train))
    plt.plot([0, 1])
    plt.plot(base_fpr, base_tpr)
    print("auc score :", auc(base_fpr, base_tpr))

    return train_auc_roc_curve


easy_lgbm = EasyEnsembleClassifier(base_estimator=LGBMClassifier(random_state=42), n_estimators=250, n_jobs=1,
                                   random_state=42, replacement=True,
                                   sampling_strategy='auto', verbose=0,
                                   warm_start=True)
easy_lgbm.fit(X_train_svm, y_train_svm)
evaluate(easy_lgbm, X_test_svm, y_test_svm)

print(classification_report(y_train_svm, easy_lgbm.predict(X_train_svm)))
print(confusion_matrix(y_train_svm, easy_lgbm.predict(X_train_svm)))
print('Recall Score = ', recall_score(y_train_svm, easy_lgbm.predict(X_train_svm)))
print('Precision Score = ', precision_score(y_train_svm, easy_lgbm.predict(X_train_svm)))

print(f1_score(y_train_svm, easy_lgbm.predict(X_train_svm)))
print(f1_score(y_test_svm, easy_lgbm.predict(X_test_svm)))

eli5_permutation = PermutationImportance(estimator=easy_lgbm, scoring='f1', random_state=42, n_iter=5)
eli5_permutation.fit(X_test_svm, y_test_svm)
eli5_permutation.feature_importances_.T.reshape(-1, 1)

feature_importance_with_eli5 = pd.DataFrame(np.hstack((np.array([X.columns[0:]]).T,
                                            eli5_permutation.feature_importances_.T.reshape(-1, 1))),
                                            columns=['feature', 'importance'])
feature_importance_with_eli5['importance'] = pd.to_numeric(feature_importance_with_eli5['importance'])
feature_importance_with_eli5.sort_values(by='importance', ascending=False)

fig = plt.figure(figsize=(15, 8))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.barplot(x='importance', y='feature', data=feature_importance_with_eli5,
            order=feature_importance_with_eli5.sort_values('importance', ascending=False).feature)


pickle.dump(easy_lgbm, open('EasyEnsembleClassifier_with_LGBMClassifier_as_base_estimator.pickle', 'wb'))
