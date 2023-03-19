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
print(knn.fit(X_train,y_train))

# Test the model on the testing set
y_pred = knn.predict(X_test)

#check accuracy of the model on the test data
knn.score(X_test, y_test)
print(knn.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=5)

#train model with cv of 5
#cv_scores = cross_val_score(knn_cv, X, y, cv=5)
cv_scores = cross_val_score(knn_cv, principal_components, y, cv=5)


#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

from sklearn.model_selection import GridSearchCV

#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}

#gridsearch to test all values for n_neighbors
#knn_gscv = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', return_train_score=False,verbose=1)
knn_gscv = GridSearchCV(knn2, param_grid, cv=5, scoring='accuracy', return_train_score=False, verbose=2)

#fit model to data
knn_gscv.fit(X, y)

# N dimensional data
ndim = 3
mu = np.array([10] * ndim) # Mean
sigma = np.zeros((ndim, ndim)) - 1.8 # Covariance
np.fill_diagonal(sigma, 3.5)
print("Mu ", mu.shape)
print("Sigma ", sigma.shape)

# Create 1000 samples using mean and sigma
org_data = rnd.multivariate_normal(mu, sigma, size=(1000))
print("Data shape ", org_data.shape)

# Subtract mean from data
mean = np.mean(org_data, axis= 0)
print("Mean ", mean.shape)
mean_data = org_data - mean
print("Data after subtracting mean ", org_data.shape, "\n")

#Compute the covariance matrix
cov = np.cov(mean_data.T)
cov = np.round(cov, 2)
print("Covariance matrix ", cov.shape, "\n")

#Perform eigen decomposition of covariance matrix
eig_val, eig_vec = np.linalg.eig(cov)
print("Eigen vectors ", eig_vec.shape)
print("Eigen values ", eig_val.shape, "\n")

# Sort eigen values and corresponding eigen vectors in descending order
indices = np.arange(0,len(eig_val), 1)
indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
eig_val = eig_val[indices]
eig_vec = eig_vec[:,indices]
print("Sorted Eigen vectors ", eig_vec.shape)
print("Sorted Eigen values ", eig_val.shape, "\n")

# Take transpose of eigen vectors with data
pca_data = mean_data.dot(eig_vec)
print("Transformed data ", pca_data.shape)

# Plot data

fig, ax = plt.subplots(1,3, figsize= (15,15))
# Plot original data
ax[0].scatter(org_data[:,0], org_data[:,1], color='blue', marker='.')

# Plot data after subtracting mean from data
ax[1].scatter(mean_data[:,0], mean_data[:,1], color='red', marker='.')

# Plot transformed data
ax[2].scatter(pca_data[:,0], pca_data[:,1], color='orange', marker='.')

# Set title
ax[0].set_title("Scatter plot of original data")
ax[1].set_title("Scatter plot of data after subtracting mean")
ax[2].set_title("Scatter plot of transformed data")

# Set x ticks
ax[0].set_xticks(np.arange(-8, 1, 8))
ax[1].set_xticks(np.arange(-8, 1, 8))
ax[2].set_xticks(np.arange(-8, 1, 8))

# Set grid to 'on'
ax[0].grid('on')
ax[1].grid('on')
ax[2].grid('on')

major_axis = eig_vec[:,0].flatten()
xmin = np.amin(pca_data[:,0])
xmax = np.amax(pca_data[:,0])
ymin = np.amin(pca_data[:,1])
ymax = np.amax(pca_data[:,1])

plt.show()
plt.close('all')

#Reverse PCA transformation
recon_data = pca_data.dot(eig_vec.T) + mean
print(recon_data.shape)

# Plot reconstructed data

fig, ax = plt.subplots(1,3, figsize= (15, 15))
ax[0].scatter(org_data[:,0], org_data[:,1], color='blue', marker='.')
ax[1].scatter(mean_data[:,0], mean_data[:,1], color='red', marker='.')
ax[2].scatter(recon_data[:,0], recon_data[:,1], color='orange', marker='.')
ax[0].set_title("Scatter plot of original data")
ax[1].set_title("Scatter plot of data after subtracting mean")
ax[2].set_title("Scatter plot of reconstructed data")
ax[0].grid('on')
ax[1].grid('on')
ax[2].grid('on')
plt.show()

#Compute reconstruction loss
loss = np.mean(np.square(recon_data - org_data))
print("Reconstruction loss ", loss)