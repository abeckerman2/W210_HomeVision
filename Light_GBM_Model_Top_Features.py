#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

# In[18]:


# Load the dataset
#data = pd.read_csv('./data_cleaned_filtered.csv')
#Andrew load
data = pd.read_csv(r'C:\Users\abeck\OneDrive\Desktop\data_cleaned_filtered_v2.csv')


# In[19]:


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


# In[20]:


#Optional - uncomment if you want to remove outliers from dataset
#Find Values with a z-score greater than a certain threshold (e.g. 4) can be considered potential outliers.

# Calculate z-scores for specific features
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

# print(total_list.index.tolist())

# df = df.drop(total_list.index.tolist())


# ### Further removing some data with extremely low/high prices

# In[21]:


percentile_10 = df['Price'].quantile(0.1)
percentile_90 = df['Price'].quantile(0.9)
df = df[(df['Price'] >= percentile_10) & (df['Price'] <= percentile_90)]


# #### Missing data

# In[22]:


# Drop features with more than 40% missing data
missing_data = df.isnull().sum() / df.shape[0]
missing_data = missing_data.sort_values(ascending=False)
print(missing_data.head(n=10))


features_to_drop = missing_data[missing_data > 0.5].index.values
df.drop(features_to_drop, axis=1, inplace=True)


# #### Baseline: excluding all the macro/regional features

# In[23]:


# Uncomment to remove macro/regional features
#column_name = '2019 GDP'
#column_index = df.columns.get_loc(column_name)
#df_baseline = df.iloc[:, :column_index]


# In[24]:


df_baseline = df


# In[25]:


from sklearn.preprocessing import LabelEncoder


# In[26]:


X = df_baseline.drop('Price', axis=1)
y = df_baseline['Price']
X_num = X.select_dtypes(include='number').copy().fillna(-9999)  # Imputing missing value as -9999
X_cat = X.select_dtypes(exclude='number').copy().apply(lambda x: LabelEncoder().fit_transform(x.astype(str)))
X_1 = pd.concat([X_num, X_cat], axis=1)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)


# In[28]:


from sklearn.ensemble import GradientBoostingRegressor


# In[29]:


#LGB Model
lgbm = lgb.LGBMRegressor()
model_1 = lgbm.fit(X_train, y_train)


# In[30]:


y_pred = model_1.predict(X_test)


# In[31]:


#Mean Error
#percentage_error = np.abs((y_test - y_pred) / y_test)
#mean_percentage_error = np.mean(percentage_error)
#print("Mean Percentage Error:", mean_percentage_error)

#Median Error
percentage_error = np.abs((y_test - y_pred) / y_test)
median_percentage_error = np.median(percentage_error)
print("Median Percentage Error:", median_percentage_error)


# ### Feature Importance

# In[20]:


#Feature Importance

feature_importance = lgbm.feature_importances_


# In[26]:


importance_df_v2 = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})


# In[27]:


importance_df_v2 = importance_df.sort_values(by='Importance', ascending=False)


# In[28]:


print(importance_df)


# In[30]:


# Export the DataFrame to a CSV file
importance_df_v2.to_csv('importance_df.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




