import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from openpyxl import *
from collections import defaultdict
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import os, pickle, operator, create_feature_vectors

create_feature_vectors.initialize_vectors(True, False, False)
feature_vectors = {}

# Original Feature Vectors
# with open('featureVectors.p') as data_file:    
#   feature_vectors = pickle.load(data_file)

# With everything but HIV rate imputed:
with open('imputed_feature_vectors_no_hiv.p') as data_file:    
  feature_vectors = pickle.load(data_file)

feature_vectors_arr = []
hiv_arr = []

for key, val in feature_vectors.iteritems():
  features, hiv = val
  if hiv is not None:
    feature_vectors_arr.append(features)
    hiv_arr.append(hiv)

vec = DictVectorizer()
data = vec.fit_transform(feature_vectors_arr).toarray()
feature_names = vec.get_feature_names()

partition = int(len(data) * .7)
training_data = preprocessing.scale(data[:partition])
test_data = preprocessing.scale(data[partition:])
hiv_training = preprocessing.scale(hiv_arr[:partition])
hiv_test = preprocessing.scale(hiv_arr[partition:])

print "%d in training and %d in test" % (len(hiv_training), len(hiv_test))
print "Feature test len: %d  hiv test len: %d" % (len(test_data), len(hiv_test))

# Fit the unweighted model
clf = linear_model.SGDRegressor()
clf.fit(training_data, hiv_training)

print feature_names
weights = []
for i, coef in enumerate(clf.coef_):
  if not "Country" in feature_names[i]:
    weights.append((feature_names[i], coef))
weights.sort(key=lambda tup: abs(tup[1]))
print weights

predictions = clf.predict(test_data)

hiv_sd = np.std(hiv_arr[:partition])
print "Standard deviation is %f" % hiv_sd
absError = mean_absolute_error(hiv_test, predictions)
medAbsErr = median_absolute_error(hiv_test, predictions)
mse  = mean_squared_error(hiv_test, predictions)
print "Mean absolute error is %f SDs.  That's %f percentage points" % (absError, absError * hiv_sd)
print "Median absolute error is %f.  That's %f percentage points" % (medAbsErr, medAbsErr * hiv_sd)
print "Mean squared error is %f.  That's %f percentage points" % (mse, mse * hiv_sd)