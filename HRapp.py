import uvicorn
from fastapi import FastAPI
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
from pydantic import BaseModel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


class FeaturesIN(BaseModel):
    city_development_index: float
    gender: str
    relevent_experience: str
    enrolled_university: str
    education_level: str
    major_discipline: str
    experience: str
    company_size: str
    company_type: str
    last_new_job: str
    training_hours: int


api = FastAPI()
    # description="city_ development _index : Developement index of the city (scaled)"
    #             "gender: Gender of candidate [Male, Female, Other]"
    #             "relevent_experience: Relevant experience of candidate [No relevent experience, Has relevent Experience]"
    #             "enrolled_university: Type of University course enrolled if any[no_enrollment, Full time course, Part time course, "
    #             "education_level: Education level of candidate [Graduate, Masters, High School, Phd, Primary School]"
    #             "major_discipline :Education major discipline of candidate [STEM, Humanities, Other, Business Degree, Arts, No Major"
    #             "experience: Total Experience of Candidate [<1 :- for less than 1 year, >20 :- for more than 20 years]"
    #             "company_size: No of employees in current employer's company [Enter in below given range : <10 for less than 10, 50-99, 100-500, 500-999, 1000-4999, 5000-9999, 10000+ for more than 10000"
    #             "company_type : Type of current employer [Pvt Ltd, Funded Startup, Public Sector, Early Stage Startup, NGO, Other]"
    #             "lastnewjob: Difference in years between previous job and current job [>4 for more than 4 years]"
    #             "training_hours: training hours completed"
try:
    pickle_in = open("EasyEnsembleClassifier_with_LGBMClassifier_as_base_estimator.pickle", "rb")
except (pickle.UnpicklingError):
    classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000

@api.post('/predict_job_change')
def predict(data1: FeaturesIN):
    data = data1.dict()
    data = pd.DataFrame(data, index=[0])
    le = LabelEncoder()
    to_labelencode = data[['city_development_index', 'gender', 'relevent_experience', 'enrolled_university',
                           'education_level', 'major_discipline', 'experience', 'company_size', 'company_type',
                           'last_new_job', 'training_hours']]
    train_temp = to_labelencode.astype("str").apply(le.fit_transform)
    train_label_encode = train_temp.where(~to_labelencode.isna(), to_labelencode)

    lr = LinearRegression()
    mice_imputer = IterativeImputer(random_state=42, estimator=lr, max_iter=10, n_nearest_features=2,
                                    imputation_order='roman')
    cleaned_train_data = mice_imputer.fit_transform(train_label_encode)

    cleaned_train_data = pd.DataFrame(cleaned_train_data)
    cleaned_train_data.columns = ['gender', 'relevent_experience', 'enrolled_university', 'education_level',
                                  'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job',
                                  'city_development_index', 'training_hours']

    temp = [cleaned_train_data['gender'], cleaned_train_data['relevent_experience'],
            cleaned_train_data['enrolled_university'], cleaned_train_data['education_level'],
            cleaned_train_data['major_discipline'], cleaned_train_data['experience'],
            cleaned_train_data['company_size'], cleaned_train_data['company_type'], cleaned_train_data['last_new_job'],
            cleaned_train_data['city_development_index'], cleaned_train_data['training_hours']]
    temp = np.array(temp)
    temp = temp.reshape(-1, 11)
    prediction = classifier.predict(temp)

    if prediction[0] > 0.5:
        prediction = "Will join the company"
    else:
        prediction = "Will not join the company"
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(api, host='127.0.0.1', port=8000)
