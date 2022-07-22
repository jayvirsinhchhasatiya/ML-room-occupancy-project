# Importing the libraries
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix
import pickle

# loading the dataset of room occupancy
occupancy_dataset= pd.read_csv('C:\\Users\\JAYVIR CHHASATIYA\\Desktop\\Data science\\case-study-june-2022 (1)\\data\\occupancy.csv')

# seperating the data and labels
X = occupancy_dataset.drop(['Occupancy','date','HumidityRatio'],axis=1)
Y = occupancy_dataset['Occupancy']

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
#stratify for splitting dataset which will not led to move all 1 output in X and 0 in Y it will be equal

classifier_rfc = RandomForestClassifier()

# fitting model with training data
classifier_rfc.fit(X_train,Y_train)

# saving model to disk
pickle.dump(classifier_rfc, open('model.pkl','wb'))

# loding model to compare the results
model = pickle.load(open('model.pkl','rb'))


