#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!  
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
predict = clf.predict(X_test)

print accuracy_score(y_test,predict)

