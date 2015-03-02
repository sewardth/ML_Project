#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import linear_model


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
del data_dict['TOTAL']
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

new_dict =[(x,data_dict[x].get('salary',0) + data_dict[x].get('bonus',0)) for x in data_dict if data_dict[x].get('salary',0) != 'NaN' and data_dict[x].get('bonus',0)!='NaN']
print sorted(new_dict,key = lambda x:x[1], reverse=True)


# maxSalary =0
# maxBonus =0
# for x in data:
# 	if x[0]>maxSalary: maxSalary = x[0]
# 	if x[1]>maxBonus: maxBonus =x[1]


# for x in data_dict:
# 	if data_dict[x]['bonus'] == maxBonus:
# 		print x
