import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from openpyxl import *
from collections import defaultdict
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import os, pickle, operator
import country_mappings
import impute, create_feature_vectors
import json
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner

# This next line creates the feature vectors from scratch with imputation.  To get rid of imputation, pass in False as the first param.  To use latitude and longitude, pass in True as the second param.  To exclude HIV, pass in False as third param
create_feature_vectors.initialize_vectors(True, True, False)

feature_vectors = {}
with open('imputed_feature_vectors.p') as data_file:    
  feature_vectors = pickle.load(data_file)

feature_vectors_arr = []
hiv_arr = []

for key, val in feature_vectors.iteritems():
  features, hiv = val
  if hiv is not None:
    feature_vectors_arr.append(features)
    hiv_arr.append(hiv)

# Let's first prepare some data!
bayes_partition = partition = int(len(feature_vectors_arr) * .7)
training_arr = feature_vectors_arr[:partition]
test_arr = feature_vectors_arr[partition:]
hiv_training_arr = hiv_arr[:partition]
hiv_test_arr = hiv_arr[partition:]

print "%d elems in training_arr out of %d total" % (len(training_arr), len(feature_vectors_arr))

vertexMap = {}

condensed_feature_vectors =[]
mainFeatures = ['indicator total health expenditure perc of GDP.xlsx', 'Indicator_BMI male ASM.xlsx', 'indicator food_consumption.xlsx', 'indicator_estimated incidence infectious tb per 100000.xlsx', 'indicator life_expectancy_at_birth.xlsx', 'indicator gapminder infant_mortality.xlsx', 'lat', 'long', 'lat-squared', 'long-squared']
# To use every spreadsheet instead: mainFeatures = feature_vectors[('France', 2002)][0].keys()
# mainFeatures.remove('Country')
vertices = set(mainFeatures)
for i, sample in enumerate(training_arr):
  newSample = {}
  newSample['HIV'] = hiv_training_arr[i]
  for k in sample.keys():
    if k in vertices:
      newSample[k] = sample[k]
      if vertexMap.get(k) == None:
        vertexMap[k] = [sample]
      else:
        vertexMap[k].append(sample)
  condensed_feature_vectors.append(newSample)

# Instantiate and run learner 
learner = PGMLearner()
bayesianNetwork = learner.lg_estimatebn(condensed_feature_vectors)
print json.dumps(bayesianNetwork.Vdata, indent=2)
print json.dumps(bayesianNetwork.E, indent=2)

# Evaluation
predictions = []
for sample in test_arr:
  predictionSamples = bayesianNetwork.randomsample(500, sample)
  hivPredictionSamples = [pSample['HIV'] for pSample in predictionSamples]
  predictions.append(np.mean(hivPredictionSamples))

result_pairs = [(predictions[i], hiv_test_arr[i]) for i in range(len(hiv_test_arr))]

predictions_std = preprocessing.scale(predictions)

# # With standardized predictions, SD = 1, so we're just getting how many SDs off we are
# test_sd = np.std(hiv_test)
# print "Sd is %f" % test_sd

# Without normalizing predictions, we need to use hiv_test_arr because that has the unnormalized HIV rates.  Now the error units are HIV rate percentage points.
print "Mean absolute error is %f" % mean_absolute_error(hiv_test_arr, predictions)
print "Median absolute error is %f" % median_absolute_error(hiv_test_arr, predictions) 
print "Mean squared error is %f" % mean_squared_error(hiv_test_arr, predictions) 

