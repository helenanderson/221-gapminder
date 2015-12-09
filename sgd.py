import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from openpyxl import *
from collections import defaultdict
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import os, pickle, operator


feature_vectors = {}

# Original Feature Vectors
# with open('featureVectors.p') as data_file:    
#   feature_vectors = pickle.load(data_file)

# With everything but HIV rate imputed
with open('imputed_feature_vectors.p') as data_file:    
  feature_vectors = pickle.load(data_file)

feature_vectors_arr = []
hiv_arr = []

for key, val in feature_vectors.iteritems():
  features, hiv = val
  if hiv is not None:
    feature_vectors_arr.append(features)
    hiv_arr.append(hiv)
  #   if cur < partition:
  #     feature_vectors_training.append(features)
  #     hiv_training.append(hiv)
  #   else:
  #     feature_vectors_test.append(features)
  #     hiv_test.append(hiv)
  # cur += 1

# print feature_vectors_test

vec = DictVectorizer()
data = vec.fit_transform(feature_vectors_arr).toarray()
feature_names = vec.get_feature_names()
# data = preprocessing.scale(data)
# hiv_arr = preprocessing.scale(hiv_arr)

partition = int(len(data) * .7)
training_data = preprocessing.scale(data[:partition])
test_data = preprocessing.scale(data[partition:])
hiv_training = preprocessing.scale(hiv_arr[:partition])
hiv_test = preprocessing.scale(hiv_arr[partition:])

# # print training_data[0]
# # print hiv_training
print "%d in training and %d in test" % (len(hiv_training), len(hiv_test))
print "Feature test len: %d  hiv test len: %d" % (len(test_data), len(hiv_test))

# print vec.get_feature_names()

# # we create 20 points
# np.random.seed(0)
# # X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
# X = [[3, 1], [2, 1], [1, 1], [-1, -2], [-1, -3], [-4, -5]]
# y = [3, 2, 1, -1, -2, -3]
# sample_weight = 100 * np.abs(np.random.randn(20))
# # and assign a bigger weight to the last 10 samples
# sample_weight[:10] *= 10

# print X
# # print Xtest
# print y

# X = [[1, 1], [2, 2], [3, 3], [4, 4]]
# Y = [1, 2, 3, 4]
# p = [[1, 1], [2, 2]]
# e = [1, 2]

# fit the unweighted model
clf = linear_model.SGDRegressor()
clf.fit(training_data, hiv_training)


print feature_names
# print clf.coef_
weights = []
for i, coef in enumerate(clf.coef_):
  if not "Country" in feature_names[i]:
    weights.append((feature_names[i], coef))
weights.sort(key=lambda tup: abs(tup[1]))
print weights

# print predictions

# totalError = float(0)
# for i, entry  in enumerate(predictions):
#   # print np.asscalar(entry)
#   totalError += abs(np.asscalar(entry) - hiv_test[i])
#   # print totalError

predictions = clf.predict(test_data)

# avgError = totalError / len(predictions)
hiv_sd = np.std(hiv_arr[:partition])
print "Sd is %f" % hiv_sd
print mean_absolute_error(hiv_test, predictions) * hiv_sd
print median_absolute_error(hiv_test, predictions) * hiv_sd
print mean_squared_error(hiv_test, predictions) * hiv_sd
absError = mean_absolute_error(hiv_test, predictions)
medAbsErr = median_absolute_error(hiv_test, predictions)
mse  = mean_squared_error(hiv_test, predictions)
print "Mean absolute error is %f SDs.  That's %f percentage points" % (absError, absError * hiv_sd)
print "Median absolute error is %f.  That's %f percentage points" % (medAbsErr, medAbsErr * hiv_sd)
print "Mean squared error is %f.  That's %f percentage points" % (mse, mse * hiv_sd)