import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os, sys, numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor

# Load the dataset
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None)

# Set column names
column_names = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over",
                "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail", "word_freq_receive",
                "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
                "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", "word_freq_your",
                "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl",
                "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet",
                "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
                "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct", "word_freq_cs",
                "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", "word_freq_edu",
                "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
                "char_freq_$", "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
                "capital_run_length_total", "is_spam"]

# Set the column names of the dataframe
data.columns = column_names
#print dataset
print(data)
#prints information of data
data.info()

data.shape
print(data.shape)
print(data['is_spam'].value_counts())

# check the shape of the dataframe before removing duplicates
print('Before removing duplicates:', data.shape)

# remove duplicates and keep the first occurrence
data = data.drop_duplicates()

# check the shape of the dataframe after removing duplicates
print('After removing duplicates:', data.shape)

print(data.shape)
print(data['is_spam'].value_counts())

#check for missing values
print(data.isnull().sum())

corr_matrix = data.corr()
plt.figure(figsize=(10,10))
plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.show()

print(data.describe())
print(data.groupby('is_spam').mean())

# Split the data into features and labels
X = data.iloc[:, :-1]  #features
y = data.iloc[:, -1]   #Labels

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a PCA object and fit it to the standardized features
pca = PCA()
pca.fit(X_std)

# Calculate the cumulative sum of explained variance ratio
cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
exp_var = pca.explained_variance_ratio_

#Determine the number of components to keep
n_components = np.argmax(cumulative_var_ratio >= 0.95) + 1

# Apply PCA with the optimum number of components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_std)

print(n_components)

plt.plot(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

plt.plot(exp_var, marker='o')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.title('Explained variance ratio by principal component')
plt.show()

plt.plot(cumulative_var_ratio, marker='o')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative explained variance by number of principal components')
plt.show()

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(principal_components, y, test_size=0.2, random_state=42)

# Create the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
# Create the KNN model with the optimal number of neighbors
#knn = KNeighborsClassifier(n_neighbors=optimal_k)

# Train the model on the training set
knn.fit(X_train, y_train)


# Test the model on the testing set
y_pred = knn.predict(X_test)

#check accuracy of our model on the test data
knn.score(X_test, y_test)