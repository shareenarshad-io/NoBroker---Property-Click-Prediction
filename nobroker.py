#Read and Explore all Datasets

##### Import required libraries 
# Import the pandas library as pd
import pandas as pd

# Import the numpy library as np
import numpy as np

# Import the seaborn library as sns
import seaborn as sns

# Import the matplotlib.pyplot library as plt
import matplotlib.pyplot as plt

# Import the json library
import json

# View options for pandas
#pd.set_option('max_columns', 50)
#pd.set_option('max_rows', 10)

# Read all data

# Properties datas
data = pd.read_csv('/Users/shareenarshad/NoBroker---Property-Click-Prediction/property_data_set.csv',parse_dates = ['activation_date'], 
                   infer_datetime_format = True, dayfirst=True)

# Data containing the timestamps of interaction on the properties
interaction = pd.read_csv('/Users/shareenarshad/NoBroker---Property-Click-Prediction/property_interactions.csv',
                          parse_dates = ['request_date'] , infer_datetime_format = True, dayfirst=True)

# Data containing photo counts of properties
pics = pd.read_table('/Users/shareenarshad/NoBroker---Property-Click-Prediction/property_photos.tsv')

# Print shape (num. of rows, num. of columns) of all data 
print('Property data Shape', data.shape)
print('Pics data Shape',pics.shape)
print('Interaction data Shape',interaction.shape)


# Sample of property data
data.sample(2)

# Sample of pics data
pics.sample(2)

# Sample of interaction data
interaction.sample(2)

'''
Data Engineering
Handling Pics Data
'''

# Show the first five rows
pics.head()

# Types of columns
pics.dtypes

# Number of nan values
pics.isna().sum()

# Try to correct the first Json
text_before = pics['photo_urls'][0]
print('Before Correction: \n\n', text_before)
# Try to replace corrupted values then convert to json 
text_after = text_before.replace('\\' , '').replace('{title','{"title').replace(']"' , ']').replace('],"', ']","')
print("\n\nAfter correction and converted to json: \n\n", json.loads(text_after))

# Function to correct corrupted json and get count of photos
def correction (x):
    # if value is null put count with 0 photos
    if x is np.nan or x == 'NaN':
        return 0
    else :
        # Replace corrupted values then convert to json and get count of photos
        return len(json.loads( x.replace('\\' , '').replace('{title','{"title').replace(']"' , ']').replace('],"', ']","') ))
        
# Apply Correction Function
pics['photo_count'] = pics['photo_urls'].apply(correction)

# Delete photo_urls column 
del pics['photo_urls']
# Sample of Pics data
pics.sample(5)

#Number of Interaction Within 3 Days

# Merge data with interactions data on property_id
num_req = pd.merge(data, interaction, on ='property_id')[['property_id', 'request_date', 'activation_date']]
num_req.head(5)

# Get a Time between Request and Activation Date to be able to select request within the number of days
#num_req['request_day'] = (num_req['request_date'] - num_req['activation_date']) / np.timedelta64(1, 'D')


# Show the first row of data
#num_req.head(1)
'''
# Get a count of requests in the first 3 days  
num_req_within_3d = num_req[num_req['request_day'] < 3].groupby('property_id').agg({ 'request_day':'count'}).reset_index()
# Show every property id with the number of requests in the first 3 days
num_req_within_3d = num_req_within_3d.rename({'request_day':'request_day_within_3d'},axis=1)
# Dataset with the number of requests within 3 days
num_req_within_3d

num_req_within_3d['request_day_within_3d'].value_counts()[:10]

def divide(x):
    if x in [1,2]:
        return 'cat_1_to_2'
    elif x in [3,4,5]:
        return 'cat_3_to_5'
    else:
        return 'cat_above_5'

num_req_within_3d['categories_3day'] = num_req_within_3d['request_day_within_3d'].apply(divide)
num_req_within_3d.head(3)

num_req_within_3d['categories_3day'].value_counts()

#Number of Interaction Within 7 Days

# Get a count of requests in the first 7 days  
num_req_within_7d = num_req[num_req['request_day'] < 7].groupby('property_id').agg({ 'request_day':'count'}).reset_index()
# Show every property id with the number of requests in the first 7 days
num_req_within_7d = num_req_within_7d.rename({'request_day':'request_day_within_7d'},axis=1)
# Dataset with the number of requests within 7 days
num_req_within_7d

num_req_within_7d['request_day_within_7d'].value_counts()[:10]

num_req_within_7d['categories_7day'] = num_req_within_7d['request_day_within_7d'].apply(divide)
num_req_within_7d.head(3)

num_req_within_7d['categories_7day'].value_counts()

#Merge Data
data.sample()

pics.sample()

num_req_within_3d.sample()
num_req_within_7d.sample()


print(num_req_within_3d.shape)
print(num_req_within_7d.shape)

label_data = pd.merge(num_req_within_7d, num_req_within_3d, on ='property_id' , how='left')
# label_data['request_day_within_3d'] = label_data['request_day_within_3d'].fillna(0)
label_data.head(3)


label_data.isna().sum()
'''
data_with_pics = pd.merge(data, pics, on ='property_id', how = 'left')
data_with_pics.head(3)

dataset = pd.merge(data_with_pics, label_data, on ='property_id')
dataset.head(3)

dataset.isna().sum()
#Exploratory Data Analysis and Processing

# Sample of dataset
dataset.sample(3)

dataset['locality'].value_counts()
# Dropped those columns that won't have an effect on the number of requests
dataset = dataset.drop(['property_id', 'activation_date' ,'latitude', 'longitude', 'pin_code','locality'  ] , axis=1)
# Some info about all columns
print('Column : Num. of null values')
print(dict(dataset.isna().sum()))
print('\n\n')
print('Column : data type')
print(dict(dataset.dtypes))
# Show histogram of the number of requests in first 3 days
plt.figure(figsize=(10,5))
sns.histplot(dataset, x="request_day_within_3d")

plt.title('histogram of num. of requests in first 3 days')
plt.show()
sns.countplot(y=dataset.categories_3day)
plt.title('Value count for each category within 3 days')
plt.show()

# Show histogram of the number of requests in first 3 days
plt.figure(figsize=(10,5))
sns.histplot(dataset, x="request_day_within_7d")

plt.title('histogram of num. of requests in first 7 days')
plt.show()

sns.countplot(y=dataset.categories_7day)
plt.title('Value count for each category within 7 days')
plt.show()
