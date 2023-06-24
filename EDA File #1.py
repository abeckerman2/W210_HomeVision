#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install Packages
#pip install pandas
#pip install seaborn

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import *


# In[2]:


#Load Data
import pandas as pd

df = pd.read_csv(r'C:\Users\abeck\OneDrive\Desktop\data_combined_clean.csv')
df_two = pd.read_csv(r'C:\Users\abeck\OneDrive\Desktop\data_cleaned_filtered_v2.csv')
print(df.columns)


# In[3]:


# Display the first few rows
#print(df.head())

# Get the dimensions of the dataset (rows, columns)
print(df.shape)

# Summary statistics of numerical columns
print(df.describe())

# Information about columns and their data types
print(df.info())


# In[4]:


# Check for missing values
x = df.isnull().sum()

x.sort_values(ascending = False).head(20)

# Handle missing values (examples)
#df.dropna()  # Drop rows with missing values
#df.fillna(value)  # Fill missing values with a specific value

df_two['Median Household Income']


# In[5]:


import seaborn as sns

# Histogram of a numerical column
#Changing figsize
#plt.figure(figsize=(10, 5)) 
plt.hist(df['Year Built'])
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Houses built by Year')
plt.show()

# Box plot of a numerical column
sns.boxplot(x=df['Last Sold Year'])
plt.xlabel('Year')
#plt.ylabel('Values')
plt.title('Boxplot of Last Sold Year')
plt.show()

# Correlation matrix
trunc_df = df_two[['Price', 'Attic', 'Tennis Court', 'Fireplace', 'Security System',  'Refrigerator', 'Lot Size', 'Beds', 'Bath', 'Median Household Income']] 

corr_matrix = trunc_df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()


# In[6]:


# Countplot of a categorical column
sns.countplot(x='State', data=df)
plt.figure(figsize=(10, 5)) 
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Countplot')
plt.show()

# Cross-tabulation of two categorical columns
cross_tab = pd.crosstab(df['Bath'], df['Beds'])
print(cross_tab)


# In[7]:


# Scatter plot
plt.scatter(df['Beds'], df['Price'])
plt.xlabel('Beds')
plt.ylabel('Price')
plt.title('# of Beds by Price')
plt.show()

plt.scatter(df['Bath'], df['Price'])
plt.xlabel('Baths')
plt.ylabel('Price')
plt.title('# of Baths by Price')
plt.show()


# In[8]:


#State Average
state_averages = df.groupby([df['State']]).mean()
#state_averages = state_averages.reset_index(drop=True)
print(state_averages)
state_averages.index


# In[9]:


#Label Data

categories = state_averages.index
values1 = state_averages['Price']

#Changing figsize
plt.figure(figsize=(10, 5)) 

# Adding labels and title
plt.xlabel('State')
plt.ylabel('Price (in $M)')
plt.title('Average House Prices by State')

# Plotting
plt.bar(categories, values1, label='Group 1')

# Displaying the chart
plt.show()


# In[10]:


import plotly.express as px
import pandas as pd

fig = px.scatter_geo(df,lat='Latitude',lon='Longitude')
fig.update_layout(title = 'World map', title_x=0.5)
fig.show()


# In[ ]:





# In[12]:





# In[14]:





# In[ ]:





# In[ ]:





# In[ ]:




