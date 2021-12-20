import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

start = time.clock()

train_data = pd.read_csv("train.csv") #reading the csv files using pandas
y = train_data['label']
X = train_data.drop(columns = 'label')
X = X/255.0
X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 2000, train_size = 100 ,random_state = 1)

MLP = MLPClassifier(hidden_layer_sizes=(1000,500,10), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, random_state=1,
       solver='adam')

#cls = joblib.load("train_model.m")
MLP.fit(X_train,y_train)
#joblib.dump(cls,'train_model.m')
y_pred=MLP.predict(X_test)
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")
print("Classification Report:","\n",metrics.classification_report(y_test,y_pred))
end = time.clock()
print (str(end-start))