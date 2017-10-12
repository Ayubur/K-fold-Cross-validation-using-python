import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import    metrics
data_frame = pd.read_csv("marketdata.csv") # input your dataset path 
feature_column_names = [0,1,2,3]
predicted_class_name = [4]
print("Enter KFold Cross Validation number: ")
k = input()
k_fold = KFold(n_splits=k)
sum = 0
for train, test in k_fold.split(data_frame):
    x1 = np.array(data_frame.values[train])
    y1 = np.array(data_frame.values[test])
    split_test_size = len(y1)
    data = np.concatenate((x1, y1), axis=0)
    dt =  pd.DataFrame(data)
    x = dt[feature_column_names].values
    y = dt[predicted_class_name].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)
    nb_model = DecisionTreeClassifier()
    x_train = [dict(enumerate(x_trains)) for x_trains in x_train]
    x_test = [dict(enumerate(x_tests)) for x_tests in x_test]
    vect = DictVectorizer(sparse=False)
    a = vect.fit_transform(x_train)
    b = vect.transform(x_test)
    nb_model.fit(a, y_train.ravel())
    prediction = nb_model.predict(b)
    accuracy = metrics.accuracy_score(y_test, prediction)
    sum+=accuracy

print("Accuracy: ", sum/k)