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


# ### Single-Family homes only (filtered data)

# In[2]:


# Load the dataset
#data = pd.read_csv('./data_cleaned_filtered.csv')
#Andrew load
data = pd.read_csv(r'C:\Users\abeck\OneDrive\Desktop\data_cleaned_filtered_v2.csv')


# In[3]:


#Further removing some data with extremely low/high prices
percentile_10 = data['Price'].quantile(0.1)
percentile_90 = data['Price'].quantile(0.9)
data = data[(data['Price'] >= percentile_10) & (data['Price'] <= percentile_90)]


# In[4]:


#Missing data
missing_data = data.isnull().sum() / data.shape[0]
missing_data = missing_data.sort_values(ascending=False)
print(missing_data.head(n=10))


# ### Baseline Model (Predict the Price Mean)

# In[6]:


#Add variable that is the price mean for each zipcode
mean_price_by_zip = data.groupby('Zipcode')['Price'].mean()
mean_price_by_zip = pd.DataFrame(mean_price_by_zip)

data_z = data[['Price', 'Zipcode']]

#Merge mean price into dataset
merged_df = pd.merge(mean_price_by_zip, data_z, on='Zipcode')

#Create a difference variable that is the difference between predicted price and actual price
merged_df['Difference'] = merged_df['Price_x'] - merged_df['Price_y']
#Price_x is predicted
#Price_y is actual


# ### Find Median Square Error

# In[7]:


#Find Median Square Error

# Calculate the percentage error
merged_df['PercentageError'] = np.abs((merged_df['Difference']) / merged_df['Price_y']) 

# Calculate the median squared percentage error
median_squared_percentage_error = np.median(merged_df['PercentageError'])

print('Median Squared Percentage Error:', median_squared_percentage_error)


#percentage_error = np.abs((y_test - y_pred) / y_test)
#median_percentage_error = np.median(percentage_error)
#print("Median Percentage Error:", median_percentage_error)


# In[ ]:




