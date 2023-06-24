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

# In[5]:


#Add variable that is the price overall
mean_price = data['Price'].mean()

print(mean_price)

#Drop all features from dataset except price
data_z = data['Price']
data_z = pd.DataFrame(data_z)

#Merge mean price into dataset
data_z['Mean_Price'] = mean_price

#Create a difference variable that is the difference between predicted price and actual price
data_z['Difference'] = data_z['Price'] - data_z['Mean_Price']

print(data_z)


# ### Find Median Square Error

# In[8]:


#Find Median Square Error


# Calculate the percentage error
data_z['PercentageError'] = np.abs((data_z['Difference']) / data_z['Price']) 

# Calculate the median squared percentage error
median_squared_percentage_error = np.median(data_z['PercentageError'])

print('Median Squared Percentage Error:', median_squared_percentage_error)


# In[ ]:




