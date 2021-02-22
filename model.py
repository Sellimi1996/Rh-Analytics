#%% Imports
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap


from pprint import pprint
from IPython.display import display 
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score

path_file = file_path = 'C:/Users/user/Documents/HR Analytics/aug_train.csv'
df = pd.read_csv(file_path, header=0)
display(df.info(verbose = True,null_counts=True))
print(df.shape)

#%% count total number of unique values in enrollee_id column
print('Number of Unique Values: ' + str(df['enrollee_id'].nunique()))

print('Number of Unique Values: ' + str(df['city'].nunique()))
print('Number of NaN Values: ' + str(sum(df['city'].isnull())))
# top 10 cities 
print((df['city'].value_counts()[0:10]))


#%% city_development-index
print("Number of Missing Values: ", df['city_development_index'].isna().sum())
display(df['city_development_index'].describe())
boxplot = df.boxplot(column ='city_development_index')
#plt.show()


#%% gender
print("Number of Missing Values: ", df['gender'].isna().sum())
fig = px.pie(df['gender'].value_counts(), values='gender', names = df['gender'].value_counts().index,title = 'gender',template='ggplot2')
#fig.show()


#%% relevent_experience
print("Number of Missing Values: ", df['relevent_experience'].isna().sum())
fig = px.pie(df['relevent_experience'].value_counts(), values='relevent_experience', 
             names = df['relevent_experience'].value_counts().index,title = 'relevent_experience',template='ggplot2')
#fig.show()


#%% Education_level
print("Number of Missing Values: ", df['education_level'].isna().sum())
fig = px.pie(df['education_level'].value_counts(), values='education_level', 
             names = df['education_level'].value_counts().index,title = 'education_level',template='ggplot2')
#fig.show()


#%% major_discipline
print("Number of Missing Values: ", df['major_discipline'].isna().sum())
fig = px.pie(df['major_discipline'].value_counts(), values='major_discipline', 
             names = df['major_discipline'].value_counts().index,title = 'major_discipline',template='ggplot2')
#fig.show()


#%% Experience
print("Number of Missing Values: ", df['experience'].isna().sum())
fig = px.pie(df['experience'].value_counts(), values='experience', 
             names = df['experience'].value_counts().index,title = 'experience',template='ggplot2')
#fig.show()


#%% company_size
print("Number of Missing Values: ", df['company_size'].isna().sum())
fig = px.pie(df['company_size'].value_counts(), values='company_size', 
             names = df['company_size'].value_counts().index,title = 'company_size',template='ggplot2')
#fig.show()


#%% company_type
print("Number of Missing Values: ", df['company_type'].isna().sum())
fig = px.pie(df['company_type'].value_counts(), values='company_type', 
             names = df['company_type'].value_counts().index,title = 'company_type',template='ggplot2')
#fig.show()


#%% last_new_job
print("Number of Missing Values: ", df['last_new_job'].isna().sum())
fig = px.pie(df['last_new_job'].value_counts(), values='last_new_job', 
             names = df['last_new_job'].value_counts().index,title = 'last_new_job',template='ggplot2')
#fig.show()

#%% training hours
print("Number of Missing Values: ", df['training_hours'].isna().sum())
display(df['training_hours'].describe())
df.boxplot(column ='training_hours')
#plt.show()

#%% target
print("Number of Missing Values: ", df['target'].isna().sum())
fig = px.pie(df['target'].value_counts(), values='target', 
             names = df['target'].value_counts().index,title = 'target',template='ggplot2')
#fig.show()


#%%  test_database
path_file = file_path = 'C:/Users/user/Documents/HR Analytics/aug_test.csv'
df_test = pd.read_csv(file_path, header=0)
display(df_test.info(verbose = True,null_counts=True))
print(df_test.shape)



#%% Extract only the features from aug_train and aug_test and rowbind them. We then will perform label encoding so that the LightGBM can be used
#Prepare Data for LightGBM

# Seperate aug_train into target and features 
y = df['target']
X_df = df.drop('target',axis = 'columns')
# save the index for X_aug_train 
X_df_index = X_df.index.to_list()

# row bind aug_train features with aug_test features 
# this makes it easier to apply label encoding onto the entire dataset 
X_aug_total = X_df.append(df_test,ignore_index = True)
display(X_aug_total.info(verbose = True,null_counts=True))

# save the index for X_aug_test 
X_df_test_index = np.setdiff1d(X_aug_total.index.to_list() ,X_df_index)



#%% MultiColumnLabelEncoder
# Code snipet found on Stack Exchange 
# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
# from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                # convert float NaN --> string NaN
                output[col] = output[col].fillna('NaN')
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

# store the catagorical features names as a list      
cat_features = X_aug_total.select_dtypes(['object']).columns.to_list()

# use MultiColumnLabelEncoder to apply LabelEncoding on cat_features 
# uses NaN as a value , no imputation will be used for missing data
X_aug_total_transform = MultiColumnLabelEncoder(columns = cat_features).fit_transform(X_aug_total)

#%% Before and After LabelEncoding
display(X_aug_total)
display(X_aug_total_transform)
##% Split X_aug_total_transform 
X_aug_train_transform = X_aug_total_transform.iloc[X_df_index, :]
X_aug_test_transform = X_aug_total_transform.iloc[X_df_test_index, :].reset_index(drop = True)
##% Before and After LabelEncoding for aug_train 
display(X_df)
display(X_aug_train_transform)


#%%Before and After LabelEncoding for aug_test
display(df_test)
display(X_aug_test_transform)



#%%  train-test stratified split using a 80-20 split
# drop enrollee_id for aug_train as it is a useless feature 
train_x, valid_x, train_y, valid_y = train_test_split(X_aug_train_transform.drop('enrollee_id',axis = 'columns'), y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)

# Create the LightGBM data containers
# Make sure that cat_features are used
train_data=lgb.Dataset(train_x,label=train_y, categorical_feature = cat_features)
valid_data=lgb.Dataset(valid_x,label=valid_y, categorical_feature = cat_features)

#Select Hyper-Parameters
params = {'objective':'binary',
          'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}





#%% Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds= 30,
                 verbose_eval= 10
                 )

#%% Overall AUC
y_hat = lgbm.predict(X_aug_train_transform.drop('enrollee_id',axis = 'columns'))
score = roc_auc_score(y, y_hat)
print("Overall AUC: {:.3f}" .format(score))





#%% ROC Curve for training/validation data
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
y_probas = lgbm.predict(valid_x) 
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(valid_y, y_probas)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for training data')
plt.legend(loc="lower right")
plt.show()




##% Feature Importance 
lgb.plot_importance(lgbm)
plt.show()


##% Feature Importance using shap package 
lgbm.params['objective'] = 'binary'
shap_values = shap.TreeExplainer(lgbm).shap_values(valid_x)
shap.summary_plot(shap_values, valid_x)


#%% Predictions for df_test.csv
predict = lgbm.predict(X_aug_test_transform.drop('enrollee_id',axis = 'columns')) 
submission = pd.DataFrame({'enrollee_id':X_aug_test_transform['enrollee_id'],'target':predict})
display(submission)

submission.to_csv('submission.csv',index=False)