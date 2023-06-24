#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import lightgbm as lgb


# ### Single-Family homes only (filtered data)

# In[13]:


# Load the dataset
#data = pd.read_csv('./data_cleaned_filtered.csv')
#Andrew load
data = pd.read_csv(r'C:\Users\abeck\OneDrive\Desktop\data_cleaned_filtered_v2.csv')
len(data)


# In[14]:


#only keep features that have a feature importance above 100
#Median Percentage Error: 0.10915
selected_features = ['Sqr Ft',
'Lot Size',
'Year Built',
'Last Tax Assestment',
'Last Sold For',
'Days On Trulia',
'Monthly_HOA',
'Last Sold Year',
'Floors',
'Rent',
'Beds',
'Bath',
'Rooms',
'Exterior',
'Inventory',
'Median Household Income',
'Parking Spaces',
'2019 GDP change',
'RESIDUAL2019',
'Parking',
'Stories',
'NPOPCHG_2019',
'Unemployed',
'Roof',
'Unemployment rate',
'POPESTIMATE2019',
'2019 GDP',
'INTERNATIONALMIG2019',
'Heating Fuel',
'DOMESTICMIG2019',
'NATURALINC2019',
'Heating',
'Luxury',
'Force',
'Refrigerator',
'Last Tax Year',
'Dishwasher',
'Fireplace',
'Porch',
'Deck', 
'Price']

df = data[selected_features]


# In[15]:


#Optional - uncomment if you want to remove outliers from dataset
#Find Values with a z-score greater than a certain threshold (e.g. 4) can be considered potential outliers.

# # Calculate z-scores for specific features
# df['feature1_zscore'] = np.abs((df['Sqr Ft'] - df['Sqr Ft'].mean()) / df['Sqr Ft'].std())
# df['feature2_zscore'] = np.abs((df['Lot Size'] - df['Lot Size'].mean()) / df['Lot Size'].std())
# df['feature3_zscore'] = np.abs((df['Last Sold For'] - df['Last Sold For'].mean()) / df['Last Sold For'].std())
# df['feature4_zscore'] = np.abs((df['Last Tax Assestment'] - df['Last Tax Assestment'].mean()) / df['Last Tax Assestment'].std())

# # Set a threshold for identifying outliers
# threshold = 4

# # Find outliers for specific features
# outliers_feature1 = df[df['feature1_zscore'] > threshold]
# outliers_feature2 = df[df['feature2_zscore'] > threshold]
# outliers_feature3 = df[df['feature3_zscore'] > threshold]
# outliers_feature4 = df[df['feature4_zscore'] > threshold]

# total_list = outliers_feature1 + outliers_feature2 + outliers_feature3 + outliers_feature4

# df = df.drop(total_list.index.tolist())
# print(len(df))


# ### Further removing some data with extremely low/high prices

# In[16]:


percentile_10 = df['Price'].quantile(0.1)
percentile_90 = df['Price'].quantile(0.9)
df = df[(df['Price'] >= percentile_10) & (df['Price'] <= percentile_90)]


# #### Create Model

# In[17]:


df_baseline = df


# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


X = df_baseline.drop('Price', axis=1)
y = df_baseline['Price']
X_num = X.select_dtypes(include='number').copy().fillna(-9999)  # Imputing missing value as -9999
X_cat = X.select_dtypes(exclude='number').copy().apply(lambda x: LabelEncoder().fit_transform(x.astype(str)))
X_1 = pd.concat([X_num, X_cat], axis=1)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)


# In[21]:


from sklearn.ensemble import GradientBoostingRegressor


# In[22]:


# Convert the training and testing sets into DMatrix objects
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

params = {
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
}

num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)



# In[23]:


y_pred = model.predict(dtest)


# In[24]:


#Mean Error
#percentage_error = np.abs((y_test - y_pred) / y_test)
#mean_percentage_error = np.mean(percentage_error)
#print("Mean Percentage Error:", mean_percentage_error)

#Median Error
percentage_error = np.abs((y_test - y_pred) / y_test)
median_percentage_error = np.median(percentage_error)
print("Median Percentage Error:", median_percentage_error)


# ### Feature Importance

# In[13]:


#Feature Importance
feature_importance = model.get_score(importance_type='weight')


# In[14]:


import matplotlib.pyplot as plt

feature_names = list(feature_importance.keys())
importance_scores = list(feature_importance.values())

plt.barh(range(len(importance_scores)), importance_scores, tick_label=feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance')
plt.show()

