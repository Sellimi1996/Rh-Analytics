# Rh-Analytics

Task Details:

This dataset is designed to understand the factors that lead to a person to work for a different company(leaving current job), by model(s) that uses the current credentials/demographics/experience to predict the probability of a candidate to look for a new job or will work for the company.
The whole data divided to train and test. Sample submission has been provided correspond to enrollee id of test set (enrolle id | target)

Notes:

The dataset is imbalanced.
Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality.
Missing imputation can be a part of your pipeline as well.

Features:

enrollee_id : Unique ID for candidate
city: City code
city_development_index : Developement index of the city (scaled)
gender: Gender of candidate
relevent_experience: Relevant experience of candidate
enrolled_university: Type of University course enrolled if any
education_level: Education level of candidate
major_discipline :Education major discipline of candidate
experience: Candidate total experience in years
company_size: No of employees in current employer's company
company_type : Type of current employer
lastnewjob: Difference in years between previous job and current job
training_hours: training hours completed
target: 0 – Not looking for job change, 1 – Looking for a job change

The file model.py give 84% accuracy with ML algorithim LightGBM.

Conclusion:

LightGBM is a great ML algorithim that handles catagorical features and missing values
This is a great dataset to work on and lots of knowledge can be gain from withing with this dataset
Researching and reading other Kaggle notebooks is essential for becoming a better data scientist


Challenges:

LightGBM has many parameters and other methods that can be utilize to better tune the parameters, this is my first time using LightGBM so mistakes might have occured
Working with catagorical features is difficult, especialy when using One-Hot Encoding, this leads to a messy dataframe and longer computational. This is why I opt for Label Encoding and LightGBM


