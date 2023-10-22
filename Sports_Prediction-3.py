#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
import os
import sklearn
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split


# In[119]:


df_21 = pd.read_csv("players_21.csv", low_memory=False)


# In[120]:


df_22 = pd.read_csv("players_22.csv", low_memory=False)


# In[121]:


data21 = df_21.copy()
data22 = df_22.copy()


# In[122]:


corr_matrix = df_21.corr(numeric_only=True)
corr_matrix['overall'].sort_values(ascending=False)


# In[123]:


# demonstrate removing useless varaiables df_21
df_21.drop(['nation_jersey_number','nationality_id','club_jersey_number','club_team_id','league_level','nation_team_id', 'sofifa_id'], axis=1, inplace=True)
url_columns = [col for col in df_21.columns if col.endswith('_url')]
df_21.drop(columns=url_columns, axis=1, inplace=True)


# In[124]:


corr_matrix2 = df_22.corr(numeric_only=True)
corr_matrix2['overall'].sort_values(ascending=False)


# In[125]:


# demonstrate removing useless varaiables df_22
df_22.drop(['nation_jersey_number','nationality_id','league_level','club_jersey_number','nation_team_id','club_team_id', 'sofifa_id'], axis=1, inplace=True)
url_columns = [col for col in df_22.columns if col.endswith('_url')]
df_22.drop(columns=url_columns, axis=1, inplace=True)


# In[126]:


# Exploratory data analysis (EDA) df_21
# df_21.hist(bins=50, figsize=(20,15))
# plt.show()


# In[127]:


# Exploratory data analysis (EDA) df_22
# df_22.hist(bins=50, figsize=(20,15))
# plt.show()


# In[128]:


df_21.info()


# In[129]:


# Further removal of useless variables df_21
df_21.drop(['club_name','league_name','club_position','club_loaned_from','club_joined','club_contract_valid_until','nation_position','player_tags','player_traits','real_face','player_positions', 'goalkeeping_speed', 'dob'], axis=1, inplace=True)


# In[130]:


# Further removal of useless variables df_22
df_22.drop(['club_name','league_name','club_position','club_loaned_from','club_joined','club_contract_valid_until','nation_position','player_tags','player_traits','real_face','player_positions', 'goalkeeping_speed', 'dob'], axis=1, inplace=True)


# In[131]:


df_22.info()


# In[132]:


dropped_cols = df_21.select_dtypes(include=['object'])


df_21.drop(df_21.select_dtypes(include=['object']), axis=1, inplace=True)
df_21.info()


# In[133]:


dropped_cols2 = df_22.select_dtypes(include=['object'])


df_22.drop(df_22.select_dtypes(include=['object']), axis=1, inplace=True)
df_22.info()


# In[134]:


# Imputation for df_21
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imputed_data = imp.fit_transform(df_21)
df_21 = pd.DataFrame(imputed_data, columns=df_21.columns)
df_21


# In[135]:


df_21 = pd.concat([dropped_cols,df_21], axis=1)
df_21


# In[136]:


# Imputation for df_22
from sklearn.impute import SimpleImputer
imp2 = SimpleImputer(strategy='mean')
imputed_data2 = imp2.fit_transform(df_22)
df_22 = pd.DataFrame(imputed_data2, columns=df_22.columns)
df_22


# In[137]:


df_22 = pd.concat([dropped_cols2, df_22], axis=1)
df_22


# In[138]:


df_21.info()


# In[139]:


# To-Do
# One-hot encoding ['nationality_name','preferred_foot','work_rate','body_type']
# then solve the position ratings with stuff like "89+3"


# In[140]:


# for df_21
weird_cols = df_21.select_dtypes(include=['object']).drop(['short_name','long_name','nationality_name','preferred_foot','work_rate','body_type'], axis=1)

for col in weird_cols.columns:
    weird_cols[col] = weird_cols[col].astype(str)

weird_cols


# In[141]:


# for df_22
weird_cols2 = df_22.select_dtypes(include=['object']).drop(['short_name','long_name','nationality_name','preferred_foot','work_rate','body_type'], axis=1)

for col in weird_cols2.columns:
    weird_cols2[col] = weird_cols2[col].astype(str)

weird_cols2


# In[142]:


# df_21
def calculate_result(value):
    parts = value.replace('-', '+').split('+')
    num1 = int(parts[0])
    num2 = int(parts[1]) if len(parts) > 1 else 0
    return num1 + num2

for col in weird_cols.columns:
    weird_cols[col] = weird_cols[col].apply(calculate_result)
    
weird_cols_true = weird_cols.copy()
weird_cols_true


# In[143]:


# df_22
for col in weird_cols2.columns:
    weird_cols2[col] = weird_cols2[col].apply(calculate_result)
    
weird_cols_true2 = weird_cols2.copy()
weird_cols_true2


# In[144]:


# df_21
# correcting values like "89+3"
df_21.drop(df_21.select_dtypes(include=['object']).drop(['short_name','long_name', 'nationality_name','preferred_foot','work_rate','body_type'], axis=1), axis=1, inplace=True)
begin_cols = df_21[df_21.columns[:8].tolist()]
df_21.drop(begin_cols, axis=1, inplace=True)
df_21 = pd.concat([weird_cols_true, df_21], axis=1)
df_21 = pd.concat([begin_cols, df_21], axis=1)


# In[145]:


# df_22
# correcting values like "89+3"
df_22.drop(df_22.select_dtypes(include=['object']).drop(['short_name','long_name','nationality_name','preferred_foot','work_rate','body_type'], axis=1), axis=1, inplace=True)
begin_cols2 = df_22[df_22.columns[:8].tolist()]
df_22.drop(begin_cols2, axis=1, inplace=True)
df_22 = pd.concat([weird_cols_true2, df_22], axis=1)
df_22 = pd.concat([begin_cols2, df_22], axis=1)


# In[146]:


df_21.info()


# In[147]:


df_21['work_rate'].value_counts()


# In[148]:


df_21['body_type'].value_counts()


# In[149]:


df_21['nationality_name'].value_counts()


# In[150]:


from sklearn.preprocessing import LabelEncoder


# In[151]:


# Encoding df_21
pfoot = df_21['preferred_foot']
workr = df_21['work_rate']
bodyt = df_21['body_type']

pfoot_label_encoder = LabelEncoder()
pfoot_encoded = pfoot_label_encoder.fit_transform(pfoot)
pfoot = pd.Series(pfoot_encoded, name='preferred_foot')

workr_label_encoder = LabelEncoder()
workr_encoded = workr_label_encoder.fit_transform(workr)
workr = pd.Series(workr_encoded, name='work_rate')

bodyt_label_encoder = LabelEncoder()
bodyt_encoded = bodyt_label_encoder.fit_transform(bodyt)
bodyt = pd.Series(bodyt_encoded, name='body_type')

df_21.drop(['nationality_name','preferred_foot','work_rate','body_type', 'short_name', 'long_name'], axis=1, inplace=True)

df_21 = pd.concat([df_21, pfoot, workr, bodyt], axis=1)


# In[152]:


# Encoding df_22
pfoot2 = df_22['preferred_foot']
workr2 = df_22['work_rate']
bodyt2 = df_22['body_type']

pfoot_label_encoder = LabelEncoder()
pfoot_encoded = pfoot_label_encoder.fit_transform(pfoot)
pfoot = pd.Series(pfoot_encoded, name='preferred_foot')

workr_label_encoder = LabelEncoder()
workr_encoded = workr_label_encoder.fit_transform(workr)
workr = pd.Series(workr_encoded, name='work_rate')

bodyt_label_encoder = LabelEncoder()
bodyt_encoded = bodyt_label_encoder.fit_transform(bodyt)
bodyt = pd.Series(bodyt_encoded, name='body_type')

df_22.drop(['nationality_name','preferred_foot','work_rate','body_type', 'short_name', 'long_name'], axis=1, inplace=True)

df_22 = pd.concat([df_22, pfoot2, workr2, bodyt2], axis=1)


# In[153]:


df_22.info()


# In[154]:


# FEATURE ENGINEERING df_21
correlations = df_21.corr(numeric_only=True)
correlations['overall'].sort_values(ascending=False)


# Reason for correlation selection
# negative corr()

# In[175]:


# feature subset extraction
columns_with_high_correlation = correlations['overall'].sort_values(ascending=False)[correlations['overall'] > 0.45].index.tolist()
selected_cols = [col for col in columns_with_high_correlation if col not in weird_cols_true.columns]
print(len(selected_cols))


# In[176]:


# FEATURES WITH BETTER CORRELATION greater than specified df_21
print(selected_cols)


# In[177]:


# FEATURE ENGINEERING df_22
correlations2 = df_22.corr(numeric_only=True)
correlations2['overall'].sort_values(ascending=False)


# In[178]:


# feature subset extraction
columns_with_high_correlation2 = correlations2['overall'].sort_values(ascending=False)[correlations2['overall'] > 0.45].index.tolist()
selected_cols2 = [col for col in columns_with_high_correlation2 if col not in weird_cols_true2.columns]
print(len(selected_cols2))


# In[179]:


# FEATURES WITH BETTER CORRELATION greater than specified df_22
print(selected_cols2)


# In[180]:


# SELECTED FEATURES FOR TRAINING AND TESTING
df_21 = df_21[selected_cols]


# In[181]:


df_22 = df_22[selected_cols]


# In[182]:


df_21.head()


# In[183]:


# dependent and independent variables df_21
y21 = df_21['overall']
X21 = df_21.drop('overall', axis=1)


# In[184]:


# SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X21.copy())
X21=pd.DataFrame(scaler.transform(X21.copy()), columns=X21.columns)


# In[236]:


#Saving the scaler
import pickle
filename = 'scaling2.pkl'
pickle.dump(scaler, open(filename, 'wb'))


# In[185]:


# CROSS-VALIDATION
# KFold
from sklearn.model_selection import GridSearchCV, KFold


# In[186]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X21, y21):
    print(train_index, test_index)
    Xtrain = pd.DataFrame(X21, index=train_index)
    Xtest = pd.DataFrame(X21, index=test_index)
    Ytrain = pd.DataFrame(y21, index=train_index)
    Ytest = pd.DataFrame(y21, index=test_index)


# In[187]:


Xtrain.head()


# In[188]:


Ytrain.head()


# In[189]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[190]:


# 3 models before grid search


# In[191]:


from sklearn.ensemble import RandomForestRegressor


# In[192]:


rf = RandomForestRegressor(n_estimators=50, max_depth=10, criterion='absolute_error')
rf.fit(Xtrain, Ytrain)
y_pred = rf.predict(Xtest)


# In[193]:


mean_absolute_error(Ytest, y_pred)


# In[194]:


np.sqrt(mean_squared_error(Ytest, y_pred))


# In[174]:


from sklearn.ensemble import GradientBoostingRegressor


# In[200]:


gboost = GradientBoostingRegressor(init=rf, n_estimators=100, learning_rate=0.001, criterion='friedman_mse')
gboost.fit(Xtrain, Ytrain)
y_pred=gboost.predict(Xtest)


# In[201]:


mean_absolute_error(Ytest, y_pred)


# In[202]:


np.sqrt(mean_squared_error(Ytest, y_pred))


# In[204]:


# !pip install xgboost


# In[205]:


from xgboost.sklearn import XGBRegressor


# In[213]:


xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.00001,
    objective='reg:squarederror'
)
xgb.fit(Xtrain, Ytrain)
y_pred=xgb.predict(Xtest)


# In[214]:


mean_absolute_error(Ytest, y_pred)


# In[215]:


np.sqrt(mean_squared_error(Ytest, y_pred))


# In[216]:


# GRID SEARCH


# In[221]:


param_grid = {
    'n_estimators': [30, 40, 50, 100],
    'max_depth': [10, 15, 20],
    'criterion': ['absolute_error']
}


# In[222]:


rf_grid_search = GridSearchCV(rf, param_grid, cv=kf, scoring='neg_mean_absolute_error')


# In[223]:


rf_grid_search.fit(X21, y21)


# In[224]:


y_pred=rf_grid_search.best_estimator_.predict(Xtest)


# In[225]:


mean_absolute_error(Ytest, y_pred)


# In[226]:


np.sqrt(mean_squared_error(Ytest, y_pred))


# In[227]:


#Saving A Model
filename = 'rf_fifa_model2.pkl'
pickle.dump(rf_grid_search, open(filename, 'wb'))


# In[228]:


best_params = rf_grid_search.best_params_
best_model = rf_grid_search.best_estimator_
print(best_params)
print(best_model)


# In[229]:


# EVALUATE df_22 (unseen data)
y22 = df_22['overall']
X22 = df_22.drop('overall', axis=1)


# In[230]:


# scaling independent variable of df_22
X22 = pd.DataFrame(scaler.transform(X22.copy()), columns=X22.columns)


# In[233]:


Xtest_22 = X22
Ytest_22 = y22
y_pred = rf_grid_search.best_estimator_.predict(Xtest_22)


# In[234]:


mean_absolute_error(Ytest_22, y_pred)


# In[235]:


np.sqrt(mean_squared_error(Ytest_22, y_pred))


# In[ ]:




