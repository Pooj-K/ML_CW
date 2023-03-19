# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

print(data)

# check the shape of the dataframe before removing duplicates
print('Before removing duplicates:', data.shape)

# remove duplicates and keep the first occurrence
data = data.drop_duplicates()

# check the shape of the dataframe after removing duplicates
print('After removing duplicates:', data.shape)

data.info()

#cleaning the data set
print(data.isnull().sum())

# Split the dataset into features and target variable
X = data.drop(['is_spam'], axis=1)
y = data['is_spam']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Print the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Check if either X_train or y_train is empty or None
print("X_train:", X_train)
print("y_train:", y_train)

# Train a decision tree classifier
#dtc = DecisionTreeClassifier(random_state=42)
#dtc.fit(X_train, y_train)

clf = DecisionTreeClassifier(random_state=42,criterion='gini')
clf.fit(X_train,y_train)

predictions_test=clf.predict(X_test)
# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#Checking accuracy of training dataset
predictions_train = clf.predict(X_train)
print(accuracy_score(y_train,predictions_train))

#Visualizing final decision tree
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)
plt.show()

#Evaluating test dataset
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions_test))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

#Evaluating training dataset
print(classification_report(y_train,predictions_train))
cm_train = confusion_matrix(y_train, predictions_train)
print("Confusion matrix for training corpus:\n", cm_train)

#Finding false positive rate and true positive rate
from sklearn.metrics import roc_curve,auc
dt_probs = clf.predict_proba(X_test)[:,1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test,dt_probs)

#Plotting ROC curve for our Decision Tree
auc_score_dt = auc(fpr_dt,tpr_dt)
print(auc_score_dt)

def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(10,8))
    plt.plot(fpr_dt, tpr_dt, color='orange', label='AUC = %0.2f' % auc_score_dt)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

plot_roc_curve(fpr_dt,tpr_dt)

#Pruning of the decision tree
# Determine the range of ccp_alpha values to explore
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

for ccp_alpha, impurity in zip(ccp_alphas, impurities):
    print("ccp_alpha: {:.4f}   Total impurity: {:.4f}".format(ccp_alpha, impurity))

#Visualizing alpha w.r.t impurity of leaves
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
