#!/usr/bin/env python
# coding: utf-8

# In[114]:


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

# In[115]:


# Load the dataset
#data = pd.read_csv('./data_cleaned_filtered.csv')
#Andrew load
data = pd.read_csv(r'C:\Users\abeck\OneDrive\Desktop\data_cleaned_filtered_v2.csv')


# In[116]:


# Drop all the columns showing image url
image_cols = data.filter(regex='Image')
data = data.drop(columns=image_cols)


# In[117]:


# Perform one-hot encoding on city and state variables
encoder = OneHotEncoder(sparse=False, drop='first')

encoded_data = encoder.fit_transform(data[['State', 'Home Type', 'City', 'County']])

# Create a new DataFrame with encoded city and state variables
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['State', 'Home Type', 'City', 'County']))

# Concatenate the encoded DataFrame with the original DataFrame
df_encoded = pd.concat([data, encoded_df], axis=1)

#Keeping city and state bc one hot encoded
data = df_encoded


# In[118]:


# Drop other columns not going to be used

columns_to_drop = ['Description', 'Longitude', 'Latitude', 'City','State','Zipcode', 'Address Full', 'Home Type', 'Home_ID', 
                  'County']
df = data.drop(columns=columns_to_drop)


# In[119]:


#Drop rows that are outliers in terms of Sqr Ft, Lot Size, and/or Median Household Income

# df = df.drop(index=[18825,
# 2734,
# 14940,
# 16434,
# 219,
# 843,
# 1515,
# 4273,
# 5073,
# 5375,
# 5784,
# 6254,
# 6435,
# 7106,
# 8048,
# 9555,
# 10317,
# 12331,
# 12739,
# 14889,
# 14963,
# 15663,
# 15941,
# 16377,
# 16755,
# 16965,
# 17292,
# 18105,
# 18270,
# 18449])


# ### Further removing some data with extremely low/high prices

# In[120]:


percentile_10 = df['Price'].quantile(0.1)
percentile_90 = df['Price'].quantile(0.9)
df = df[(df['Price'] >= percentile_10) & (df['Price'] <= percentile_90)]


# #### Missing data

# In[121]:


# Drop features with more than 40% missing data
missing_data = df.isnull().sum() / df.shape[0]
missing_data = missing_data.sort_values(ascending=False)
print(missing_data.head(n=10))


features_to_drop = missing_data[missing_data > 0.5].index.values
df.drop(features_to_drop, axis=1, inplace=True)


# #### Baseline: excluding all the macro/regional features

# In[122]:


# Uncomment to remove macro/regional features
#column_name = '2019 GDP'
#column_index = df.columns.get_loc(column_name)
#df_baseline = df.iloc[:, :column_index]


# In[123]:


df_baseline = df


# In[124]:


from sklearn.preprocessing import LabelEncoder


# In[125]:


X = df_baseline.drop('Price', axis=1)
y = df_baseline['Price']
X_num = X.select_dtypes(include='number').copy().fillna(-9999)  # Imputing missing value as -9999
X_cat = X.select_dtypes(exclude='number').copy().apply(lambda x: LabelEncoder().fit_transform(x.astype(str)))
X_1 = pd.concat([X_num, X_cat], axis=1)


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)


# In[127]:


from sklearn.ensemble import GradientBoostingRegressor


# In[134]:


# Convert the training and testing sets into DMatrix objects
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

params = {
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    #'lambda': 100,  # L2 regularization term (Ridge)
    #'alpha': 200  # L1 regularization term (Lasso)
}

num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)



# In[135]:


y_pred = model.predict(dtest)


# In[136]:


#Mean Error
#percentage_error = np.abs((y_test - y_pred) / y_test)
#mean_percentage_error = np.mean(percentage_error)
#print("Mean Percentage Error:", mean_percentage_error)

#Median Error
percentage_error = np.abs((y_test - y_pred) / y_test)
median_percentage_error = np.median(percentage_error)
print("Median Percentage Error:", median_percentage_error)


# In[137]:


### Feature Importance


# In[138]:


#Feature Importance
feature_importance = model.get_score(importance_type='weight')


# In[139]:


import matplotlib.pyplot as plt

feature_names = list(feature_importance.keys())
importance_scores = list(feature_importance.values())

plt.barh(range(len(importance_scores)), importance_scores, tick_label=feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance')
plt.show()


# In[ ]:




