#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import json


# #json file se data import krke data frame mai jata hai idr 
# users = []
# with open('./yelp_academic_dataset_business.json', encoding = 'cp850') as fl:
#     for i, line in enumerate(fl):
#         users.append(json.loads(line))
# df = pd.DataFrame(users)
# 
# #df = df[(df.is_open == 1)]
# df = df[(df.city=='Philadelphia') & (df.is_open == 1)]
# 
# print(df.shape)

# In[57]:


df = pd.read_csv('final_dataframe newCode radius=2All.csv')


# # Geohash

# In[58]:


# Function to generate geohash with specified precision and assign unique integer IDs
def assign_geohash_ids(df, precision):
    # Create a new column 'geohash' by applying geohash.encode on latitude and longitude columns
    df['geohash'] = df.apply(lambda row: geohash.encode(row['latitude'], row['longitude'], precision=precision), axis=1)
    
    # Assign unique integer IDs to each geohash
    df['geohash_id'] = pd.factorize(df['geohash'])[0]
    
    return df


# In[59]:


import geohash

# Call the function to assign geohash IDs with precision of 6 characters
df = assign_geohash_ids(df, precision=5)

# Print the highest unique integer ID
highest_id = df['geohash_id'].max()
print("Highest Unique Integer ID:", highest_id)


# In[60]:


df.to_csv('final_dataframe newCode radius=2All.csv', index=False)


# ### Geohash Map Print

# In[55]:


import folium

# Create a map centered on the mean of coordinates
center_lat, center_long = np.mean(df[['latitude', 'longitude']], axis=0)

map_clusters = folium.Map(location=[center_lat, center_long], zoom_start=11)

# Assign colors to the clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'gray', 'pink', 'black']

# Plot the points and clusters on the map
for lat, lon, label in zip(df['latitude'], df['longitude'], df['geohash_id']):
    if label == -1:
        color = 'gray'  # Color for noise points
    else:
        color = colors[label % len(colors)]  # Assign a color to each cluster
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.3
    ).add_to(map_clusters)

# Save the map as an HTML file
map_clusters.save("cluster_map.html")

# Display the map in Jupyter notebook
map_clusters


# # DBSCAN

# In[53]:


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Assuming you have a DataFrame called 'df' with latitude and longitude columns

# Preprocess the data
X = df[['latitude', 'longitude']].values
#X = StandardScaler().fit_transform(X)

# Apply DBSCAN
epsilon = 0.004 # The maximum distance between two samples to be considered as neighbors
min_samples = 5  # The minimum number of samples in a neighborhood to form a core point
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels = dbscan.fit_predict(X)

# Print the cluster labels
print(labels)

# Count the number of clusters (excluding noise points)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Count the number of noise points
num_noise_points = list(labels).count(-1)

# Print the number of clusters and noise points
print("Number of clusters:", num_clusters)
print("Number of noise points:", num_noise_points)


# ### DBSCAN Map

# In[54]:


import folium

# Create a map centered on the mean of coordinates
center_lat, center_long = np.mean(X, axis=0)
map_clusters = folium.Map(location=[center_lat, center_long], zoom_start=11)

# Assign colors to the clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'gray', 'pink', 'black']

# Plot the points and clusters on the map
for lat, lon, label in zip(df['latitude'], df['longitude'], labels):
    if label == -1:
        color = 'gray'  # Color for noise points
    else:
        color = colors[label % len(colors)]  # Assign a color to each cluster
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.3
    ).add_to(map_clusters)

# Save the map as an HTML file
map_clusters.save("cluster_map.html")

# Display the map in Jupyter notebook
map_clusters


# In[69]:


df.info()

del df['Business Name']
del df['City']
del df['State']
del df['latitude']
del df['longitude']
del df['geohash']


# # K-means

# In[70]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#df = pd.read_csv('final_dataframe newCode radius=2All.csv')

# Concatenate latitude and longitude columns into a new DataFrame
coordinates = df[['latitude', 'longitude']]

# Specify the number of clusters (k) and initialize the KMeans model
#k = int(highest_id*0.1)
k=2
kmeans = KMeans(n_clusters=k, init='k-means++')

# Fit the model to the coordinates
kmeans.fit(df)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Add the cluster labels as a new column to the DataFrame
df['kmeans_cluster'] = cluster_labels


# In[56]:


import folium

# Create a map centered on the mean of coordinates
center_lat, center_long = np.mean(df[['latitude', 'longitude']], axis=0)

map_clusters = folium.Map(location=[center_lat, center_long], zoom_start=11)

# Assign colors to the clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'gray', 'pink', 'black']

# Plot the points and clusters on the map
for lat, lon, label in zip(df['latitude'], df['longitude'], df['kmeans_cluster']):
    if label == -1:
        color = 'gray'  # Color for noise points
    else:
        color = colors[label % len(colors)]  # Assign a color to each cluster
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.3
    ).add_to(map_clusters)

# Save the map as an HTML file
map_clusters.save("cluster_map.html")

# Display the map in Jupyter notebook
map_clusters

