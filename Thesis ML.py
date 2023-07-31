#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.ensemble import VotingRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


RegressionData = pd.read_csv('final_dataframe newCode radius=2All.csv')
#RegressionData = pd.read_csv('output.csv')

RegressionData


# In[3]:


RegressionData.describe()


# In[4]:


#RegressionData = RegressionData.drop(RegressionData[RegressionData['Same Category Business Count'] < 2].index)
#RegressionData = RegressionData.drop(RegressionData[RegressionData['Surrounding Business Count of Different Category'] < 10].index)
#RegressionData = RegressionData.drop(RegressionData[RegressionData['Review Count'] < 10].index)


# In[5]:


RegressionData.info()


# In[6]:


import copy

data = copy.deepcopy(RegressionData)
del data['Business Name']
del data['City']
del data['State']
del data['latitude']
del data['longitude']
del data['geohash']
del data['Business Rating']

data.info()


# In[ ]:


'Average rating in radius','DiffCatWeightedMean','Average rating in radius of same category','SameCatWeightedMean','weighted_mean_SameSubCat'


# del data['dbscan_cluster_label']
# del data['kmeans_cluster_label']
# del data['neighborhood_id']
# del data['WiFi']
# del data['BikeParking']
# del data['GoodForKids']
# del data['RestaurantsGoodForGroups']
# del data['HasTV']
# del data['RestaurantsReservations']
# del data['RestaurantsPriceRange2']
# del data['OutdoorSeating']
# del data['RestaurantsDelivery']
# del data['RestaurantsTakeOut']
# del data['BusinessAcceptsCreditCards']
# del data['Review Count']

# In[7]:


plt.figure(figsize = (12 ,6))
#sns.histplot(data = data, x = data['SameCatWeightedMean'], kde = True)
sns.histplot(data = data, x = data['weighted_mean_SameSubCat'], kde = True)
plt.show()


# In[8]:


plt.figure(figsize = (12 ,6))
sns.histplot(data = data, x = data['DiffCatWeightedMean'], kde = True)
#sns.histplot(data = data, x = data['RestaurantsPriceRange2'], kde = True)
plt.show()


# In[9]:


plt.figure(figsize=(12, 6))
sns.histplot(data=data[data['FoodSubCategory'] == 1], x='weighted_mean_SameSubCat', kde=True)
plt.show()

average = data[data['FoodSubCategory'] == 0]['weighted_mean_SameSubCat'].mean()
print(average)


# In[10]:


data.sample()


# In[11]:


# Define a function to apply the transformation
def transform_rating_class(x):
    if x >= 4:
        return 2
    elif x >= 3:
        return 1
    else:
        return 0

# Apply the function to the rating class column
data['Rating Class'] = data['Business Rating'].apply(transform_rating_class)


# In[28]:


# Specify the columns you want to keep
columns_to_keep = ['Rating Class','Average rating in radius','DiffCatWeightedMean','Average rating in radius of same category','SameCatWeightedMean','weighted_mean_SameSubCat']

# Select the specified columns from the DataFrame
subset = data[columns_to_keep]

# Create the correlation heatmap with blues color palette
plt.figure(figsize=(10, 5))
sns.heatmap(subset.corr(), annot=True, cmap='Blues')
plt.show()


# In[29]:


# Specify the columns you want to keep
columns_to_keep = ['Rating Class','WiFi','BusinessAcceptsCreditCards','RestaurantsTakeOut','RestaurantsDelivery','OutdoorSeating','RestaurantsPriceRange2','RestaurantsReservations','HasTV','RestaurantsGoodForGroups','GoodForKids','BikeParking','Parking']

# Select the specified columns from the DataFrame
subset = data[columns_to_keep]

# Create the correlation heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(subset.corr(), annot=True, cmap='Blues')
plt.show()


# In[13]:


plt.figure(figsize = (40, 20))

sns.heatmap(data.corr(), annot = True)
plt.show()


# In[14]:


#del data['kmeans_cluster_label']
#del data['neighborhood_id']
del data['Business Rating']
#del data['WiFi']
#del data['Min rating in radius']
#del data['Min rating in radius of same category']


# In[15]:


# Separate Data columns and Target column

#Y = np.array(data['Business Rating'])
Y = np.array(data['Rating Class'])

#X = copy.deepcopy(data.drop('Business Rating', axis=1).copy())
X = copy.deepcopy(data.drop('Rating Class', axis=1).copy())

#print X


# ## Split Data

# In[16]:


# Split Data in 75:25 Ratio of Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=412, stratify=Y)

print ("Shape of Training Data: \n" ,"Train X: ",X_train.shape,"\nTrain Y: " ,y_train.shape)
print ("\nShape of Testing Data: \n" ,"Train X: ",X_test.shape,"\nTrain Y: " ,y_test.shape)


# # Machine Learning Model Selection and Training

# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=2000, random_state=2021)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = rf_classifier.predict(X_test)

# Compute the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Random Forest Classifier: {:.2f}%".format(accuracy*100))

# Calculate precision, recall, and F1 score
classification_metrics = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", classification_metrics)

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[18]:


# Get the feature importances
importances = rf_classifier.feature_importances_

# Print the feature importances
for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance}")
    
# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot the feature importances as a bar plot
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Print the list of features in order of importance
print(feature_importance_df['Feature'].tolist())


# In[19]:


import pandas as pd

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Get the sorted list of features
sorted_features = feature_importance_df['Feature'].tolist()

# Create the LaTeX table code
latex_table = feature_importance_df.to_latex(index=False)

# Print the sorted list and the LaTeX table
print("Sorted Features:")
print(sorted_features)
print("\nLaTeX Table:")
print(latex_table)


# In[20]:


import xgboost as xgb

# Initialize the XGBoost classifier
xgb_classifier = xgb.XGBClassifier()

# Train the classifier
xgb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = xgb_classifier.predict(X_test)

# Compute the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of XG Boost: {:.2f}%".format(accuracy*100))

# Calculate precision, recall, and F1 score
classification_metrics = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", classification_metrics)

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[21]:


from sklearn import svm

# Create an instance of the SVC classifier
clf = svm.SVC(kernel='rbf')

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Compute the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of XG Boost: {:.2f}%".format(accuracy*100))

# Calculate precision, recall, and F1 score
classification_metrics = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", classification_metrics)

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[22]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# Initialize and train the LGBMClassifier
model = LGBMClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate precision, recall, and F1 score
classification_metrics = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", classification_metrics)

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)


# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# # LazyClassifier

# In[23]:


import pandas as pd
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split

# Initialize LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit the classifier on the training data
models, predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


import matplotlib.pyplot as plt

# Extract the balanced accuracy scores from the results
balanced_accuracy = models['Balanced Accuracy']

# Generate the bar plot
balanced_accuracy.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Model')
plt.ylabel('Balanced Accuracy')
plt.title('Model Performance - Balanced Accuracy')
plt.tight_layout()

# Save the plot as an image file
plt.savefig('lazy_accuracy.png')

# Print the balanced accuracy scores
print(balanced_accuracy)


# In[ ]:


predictions

