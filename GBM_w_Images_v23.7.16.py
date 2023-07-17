import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# ### Single-Family homes only (filtered data)

# In[2]:


# Load the dataset
# data = pd.read_csv('./data_cleaned_filtered.csv')
#Dylan load
data = pd.read_csv('./data_cleaned_filtered_by_img.csv')


# In[3]:


# Drop all the columns showing image url
image_cols = data.filter(regex='Image')
data = data.drop(columns=image_cols)


# In[4]:


# Perform one-hot encoding on city and state variables
encoder = OneHotEncoder(sparse=False, drop='first')

encoded_data = encoder.fit_transform(data[['State', 'City', 'County']])

# Create a new DataFrame with encoded city and state variables
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['State', 'City', 'County']))

# Concatenate the encoded DataFrame with the original DataFrame
df_encoded = pd.concat([data, encoded_df], axis=1)

#Keeping city and state bc one hot encoded
data = df_encoded


# In[5]:


# Drop other columns not going to be used

# columns_to_drop = ['Description', 'Longitude', 'Latitude', 'City','State','Zipcode', 'Address Full', 'Home Type', 'Home_ID', 
#                   'County']
columns_to_drop = ['Description', 'Longitude', 'Latitude', 'City','State','Zipcode', 'Address Full', 'Home Type', 
                  'County']
df = data.drop(columns=columns_to_drop)


# In[6]:


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

column_name = '2019 GDP'
column_index = df.columns.get_loc(column_name)
df = df.iloc[:, :column_index]


# In[10]:

# Comment below to remove macro/regional features 
# df_baseline = df



### Image Features ###

import os
import tensorflow as tf 
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image

image_folder = "./image_data"

# Load the EfficientNet model
model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights='imagenet', include_top=False, pooling='avg')

# Define a function to extract image features using EfficientNet
def extract_image_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    features = model.predict(x)
    return features.flatten()  # Flatten the features to a 1-dimensional array

# Create a list to store image features and corresponding Home_ID
image_features_list = []
home_id_list = []

# Iterate over the dataframe and extract image features for each image
for home_id in data['Home_ID']:
    image_filename = f"{home_id}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    if os.path.exists(image_path):
        features = extract_image_features(image_path)
        image_features_list.append(features)
        home_id_list.append(home_id)
    else:
        print(f"Image not found for Home_ID: {home_id}")

# Create a new dataframe with extracted image features
image_features_df = pd.DataFrame(image_features_list, columns=[f"img_feature_{i}" for i in range(image_features_list[0].shape[0])])
image_features_df['Home_ID'] = home_id_list

# Merge the original dataframe with the image features dataframe using Home_ID
df = pd.merge(data, image_features_df, on='Home_ID')

# Save the preprocessed images for future use
preprocessed_image_folder = './preprocessed_images'
os.makedirs(preprocessed_image_folder, exist_ok=True)

for home_id in data['Home_ID']:
    image_filename = f"{home_id}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    preprocessed_image_path = os.path.join(preprocessed_image_folder, image_filename)
    if os.path.exists(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img.save(preprocessed_image_path)     
        
columns_to_drop = ['Home_ID']
df = data.drop(columns=columns_to_drop)        

# Comment below to remove image features 
df_baseline = df

## End Image Features ###



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
# Gradient Boosting model

from sklearn.ensemble import GradientBoostingRegressor


# In[15]:

GB_model_1 = GradientBoostingRegressor(max_depth=5, n_estimators=1000, learning_rate=0.1)
model_1 = GB_model_1.fit(X_train, y_train)

# In[16]:


y_pred = model_1.predict(X_test)


# In[17]:

#Median Error
percentage_error = np.abs((y_test - y_pred) / y_test)
median_percentage_error = np.median(percentage_error)
print("Median Percentage Error:", median_percentage_error)

# ### Feature Importance

# # In[20]:


# #Feature Importance

# feature_importance = lgbm.feature_importances_


# # In[26]:


# importance_df_v2 = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})


# # In[27]:


# importance_df_v2 = importance_df.sort_values(by='Importance', ascending=False)


# # In[28]:


# print(importance_df)


# # In[30]:


# # Export the DataFrame to a CSV file
# importance_df_v2.to_csv('importance_df.csv', index=False)
