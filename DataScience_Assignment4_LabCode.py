#  K-Nearest Neighbours (KNN algorithm) implementation in Scikit-Learn

import pandas as pd
# below library is used for data splitting
from sklearn.model_selection import train_test_split

# load heart dataset
df=pd.read_csv(r"C:\Users\chett\OneDrive\Desktop\heart.csv")

#Split the dataset into train (75%) and test (25%) datasets: We are considering RestingBP, Cholesterol and Max HR as the parameters that contribute to whether we have heart disease or not on the X and y axis.
X = df[['RestingBP','Cholesterol','MaxHR']]
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

#The len(X_train) will give the 75% of the total dataset result that we have 
len(X_train)
#The len(X_test) will give the 25% of the total dataset result that we have 
len(X_test)

# print 75% contents of the train dataset
X_train

#scale the features so that all of them ca be uniformly evaluated, the script for normalization is:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# train the K algorithm and make prediction out of it:
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# evaluate the matrix using confusion matrix and printing the  confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



##################################################
#  Decision tree implementation in Scikit-Learn
#Load the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#read dataset
df=pd.read_csv(r"C:\Users\chett\OneDrive\Desktop\heart.csv")

X = df[['RestingBP','Cholesterol','MaxHR']]
y = df['HeartDisease']

#split the dataset into 75% train and 25% test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
len(X_train)
len(X_test)

#decision tree classifier model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)
confusion_matrix(y_test, y_pred, labels=[0, 1])



####################################################3
# Confusion matrix for Random forest classifier
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

#Load the dataset
df=pd.read_csv(r"C:\Users\chett\OneDrive\Desktop\heart.csv")

X = df[['RestingBP','Cholesterol','MaxHR']]
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
len(X_train)
len(X_test)

# Create a Random Forest Classifier model and fit it to the training data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the testing data using the trained model
y_pred = rf_model.predict(X_test)

# Generate a confusion matrix for the predictions
cm = confusion_matrix(y_test, y_pred)
print(cm)