#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


# Load the dataset
#data = pd.read_csv('./data_cleaned_filtered.csv')
#Andrew load
data = pd.read_csv(r'C:\Users\abeck\OneDrive\Desktop\data_cleaned_filtered_v2.csv')


# In[3]:


# Drop all the columns showing image url
image_cols = data.filter(regex='Image')
data = data.drop(columns=image_cols)


# In[4]:


# Perform one-hot encoding on city and state variables
encoder = OneHotEncoder(sparse=False, drop='first')

encoded_data = encoder.fit_transform(data[['State', 'Home Type', 'City', 'County']])

# Create a new DataFrame with encoded city and state variables
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['State', 'Home Type', 'City', 'County']))

# Concatenate the encoded DataFrame with the original DataFrame
df_encoded = pd.concat([data, encoded_df], axis=1)

#Keeping city and state bc one hot encoded
data = df_encoded


# In[5]:


# Drop other columns not going to be used

columns_to_drop = ['Description', 'Longitude', 'Latitude', 'City','State','Zipcode', 'Address Full', 'Home Type', 'Home_ID', 
                  'County']
df = data.drop(columns=columns_to_drop)


# In[6]:


#Find Values with a z-score greater than a certain threshold (e.g. 4) can be considered potential outliers.

# Calculate z-scores for specific features
df['feature1_zscore'] = np.abs((df['Sqr Ft'] - df['Sqr Ft'].mean()) / df['Sqr Ft'].std())
df['feature2_zscore'] = np.abs((df['Lot Size'] - df['Lot Size'].mean()) / df['Lot Size'].std())
df['feature3_zscore'] = np.abs((df['Last Sold For'] - df['Last Sold For'].mean()) / df['Last Sold For'].std())
df['feature4_zscore'] = np.abs((df['Last Tax Assestment'] - df['Last Tax Assestment'].mean()) / df['Last Tax Assestment'].std())

# Set a threshold for identifying outliers
threshold = 4

# Find outliers for specific features
outliers_feature1 = df[df['feature1_zscore'] > threshold]
outliers_feature2 = df[df['feature2_zscore'] > threshold]
outliers_feature3 = df[df['feature3_zscore'] > threshold]
outliers_feature4 = df[df['feature4_zscore'] > threshold]

total_list = outliers_feature1 + outliers_feature2 + outliers_feature3 + outliers_feature4

print(total_list.index.tolist())

df = df.drop(total_list.index.tolist())


# ### Further removing some data with extremely low/high prices

# In[7]:


percentile_10 = df['Price'].quantile(0.1)
percentile_90 = df['Price'].quantile(0.9)
df = df[(df['Price'] >= percentile_10) & (df['Price'] <= percentile_90)]


# #### Missing data

# In[8]:


# Drop features with more than 40% missing data
missing_data = df.isnull().sum() / df.shape[0]
missing_data = missing_data.sort_values(ascending=False)
print(missing_data.head(n=10))


features_to_drop = missing_data[missing_data > 0.5].index.values
df.drop(features_to_drop, axis=1, inplace=True)


# #### Baseline: excluding all the macro/regional features

# In[9]:


# Uncomment to remove macro/regional features
#column_name = '2019 GDP'
#column_index = df.columns.get_loc(column_name)
#df_baseline = df.iloc[:, :column_index]


# In[10]:


df_baseline = df


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


X = df_baseline.drop('Price', axis=1)
y = df_baseline['Price']
X_num = X.select_dtypes(include='number').copy().fillna(-9999)  # Imputing missing value as -9999
X_cat = X.select_dtypes(exclude='number').copy().apply(lambda x: LabelEncoder().fit_transform(x.astype(str)))
X_1 = pd.concat([X_num, X_cat], axis=1)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)


# In[14]:


from sklearn.ensemble import GradientBoostingRegressor


# In[15]:


#LGB Model
lgbm = lgb.LGBMRegressor()
model_1 = lgbm.fit(X_train, y_train)


# In[16]:


y_pred = model_1.predict(X_test)


# In[17]:


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




