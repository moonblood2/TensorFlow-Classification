import tensorflow as tf
from tensorflow.contrib import learn
import sklearn
from sklearn import metrics
import numpy as np
from sklearn.cross_validation import train_test_split

iris = learn.datasets.load_iris()

x_train,x_test,y_train,y_test = train_test_split(
    iris.data,iris.target,test_size = 0.2)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
clf = learn.DNNClassifier(
    hidden_units = [10,20,10],feature_columns = feature_columns, n_classes = 3)

clf.fit(x_train,y_train, steps = 200)

score = clf.evaluate(x=x_test, y=y_test)["accuracy"]

print(score)
