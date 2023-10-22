import streamlit as st #importing streamlit
import time #importing time
import numpy as np #importing numpy
import pickle #importing pickle
import pandas as pd #importing pandas
import os #importing os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

st.title('FIFA RATING PREDICTION')


st.image(str('EASports.png'), use_column_width=True)


# Displaying a temporary welcome message
welcome_message = st.empty()
welcome_message.text("Move the sliders to match your desired ratings!")
time.sleep(1.5)#timer for the welcome message to terminate
welcome_message.empty()#code to remove the welcome message

#User inputs
feature1 = st.slider("Movement Reaction", min_value=0, max_value=100, value=50, step=1)
feature2 = st.slider("Mentality Composure", min_value=0, max_value=100, value=50, step=1)
feature3 = st.slider("Passing", min_value=0, max_value=100, value=50, step=1)
feature4 = st.slider("Potential", min_value=0, max_value=100, value=50, step=1)
feature5 = st.slider("Release Clause(Euros)", min_value=0, max_value=10000000000, value=50, step=1)
feature6 = st.slider("Dribbling", min_value=0, max_value=100, value=50, step=1)
feature7 = st.slider("Wage (Euro)", min_value=0, max_value=3000000, value=50, step=1)
feature8 = st.slider("Shot Power", min_value=0, max_value=100, value=50, step=1)
feature9 = st.slider("Value Euro", min_value=0, max_value=1000000000, value=50, step=1)
feature10 = st.slider("Mentality Vision", min_value=0, max_value=100, value=50, step=1)
feature11 = st.slider("Attacking Short Passing", min_value=0, max_value=100, value=50, step=1)
feature12 = st.slider("Physic", min_value=0, max_value=100, value=50, step=1)
feature13 = st.slider("Skill Long Passing", min_value=0, max_value=100, value=50, step=1)
feature14 = st.slider("Age", min_value=0, max_value=100, value=50, step=1)
feature15 = st.slider("Shooting", min_value=0, max_value=100, value=50, step=1)

print(feature1)


user_inputs_list = [feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,feature12,feature13,feature14,feature15]
user_inputs_list = np.array(user_inputs_list)

user_inputs = pd.DataFrame(user_inputs_list.reshape(1, -1), columns=[
    'movement_reactions','mentality_composure','passing','potential',
    'release_clause_eur','dribbling','wage_eur','power_shot_power',
    'value_eur','mentality_vision','attacking_short_passing','physic',
    'skill_long_passing','age','shooting'
])


#scaling the user inputs
loaded_scaler = pickle.load(open('scaling2.pkl', 'rb'))
scaled_user_inputs = pd.DataFrame(loaded_scaler.transform(user_inputs.copy()))


loaded_model = pickle.load(open('rf_fifa_model2.pkl', 'rb'))

prediction = loaded_model.best_estimator_.predict(scaled_user_inputs)

if st.button('SUBMIT'):
    st.write(prediction)

