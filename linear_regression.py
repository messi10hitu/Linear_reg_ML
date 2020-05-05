import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.utils import shuffle
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
# print(data)
# print(data.head(10))
# print("----------")

# This is cutting down of attributes from large no of attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head(5))
# print("--------")

# Labels are based upon the Attributes
# print(data.drop(columns="G3"))
predict = "G3"  # here predict is Label whose value we will predict from the given dataset
X = data.drop(columns=predict)  # X is our Training data
X = np.array(X)
# OR
# X = np.array(data.drop([predict], 1))  # here 1 is axis
# print(X)
# print("-------------")
Y = np.array(data[predict])  # here Y is Label and its value is we are going to predict
# print(Y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
"""
best = 0
for i in range(500):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
   
    # print(x_train)
    # print("----------")
    # print(x_test)
    # print("----------")
    # print(y_train)
    # print("----------")
    # print(y_test)
    # print("-------------------------------")

    # x_train is X which is = data.drop(columns=predict)
    # y_train is Y which is = np.array([predict])
    # x_test, y_test = This test is used to find the accuracy of our Model or Algorithm that we will create
    # test_size = 0.1 Means we are splitting 10% of our data into test samples to get the accurate results

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)  # this is used to make a straight best fit line using (x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    # why we r getting diff score at each tym bcoz our (test size = 0.1) thats why we r getting diff test data at every tym
    print(accuracy)
    if accuracy > best:
        best = accuracy
        print("----", best)
        # PICKLE MODULE = pickle is used to save our linear model
        with open('student_model.pickle', 'wb') as f:
            pickle.dump(linear, f)"""  # here we are dumping linear into the file f.

pickle_in = open('student_model.pickle', 'rb')
linear = pickle.load(pickle_in)  # here we load our pickle into our linear model = linear.fit(x_train, y_train)

# print("cofficients: ", linear.coef_)  # this is for slope
# print("intercept: ", linear.intercept_)  # this is for intercept
#
#
predictions = linear.predict(x_test)  # it comes from after training of our data linear.fit(x_train, y_train)
print(predictions)
# now we will make predictions on the x_test data
for x in range(len(predictions)):
    print("predicted: ", predictions[x], "Data: ", x_test[x], "Actual",
          y_test[x])  # 9.760309168710455 [ 8 10  1  0 12] 10  #
# p = "G1"
# style.use("ggplot")
# plt.scatter(data[p], data["G3"])
# plt.xlabel("first sem")
# plt.ylabel("final grade")
# plt.show()
