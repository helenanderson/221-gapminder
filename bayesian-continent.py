# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from openpyxl import *
from collections import defaultdict
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import os, pickle, operator
import random
import impute, create_feature_vectors
import json
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner

continents = {
  "Africa": ["Cote D'Ivoire","Mayotte",u"São Tomé and Príncipe","Congo [Republic]","Congo [DRC]",u"Côte d'Ivoire","Western Sahara","Reunion","Central African Rep.","Congo, Dem. Rep.","Algeria","Angola","Benin","Botswana","Burkina","Burkina Faso","Burundi","Cameroon","Cape Verde","Central African Republic","Chad","Comoros","Congo","Congo, Rep.","Djibouti","Egypt","Equatorial Guinea","Eritrea","Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Cote d'Ivoire","Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco","Mozambique","Namibia","Niger","Nigeria","Rwanda","Sao Tome and Principe","Senegal","Seychelles","Sierra Leone","Somalia","South Africa","South Sudan","Sudan","Swaziland","Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe"],
  "Asia": ["Myanmar [Burma]","Democratic Republic of Timor-Leste","Macao, China","Laos","Libyan Arab Jamahiriya","Korea, Dem. Rep.","Yemen, Rep.","Hong Kong, China","Syrian Arab Republic","Korea, Rep.","West Bank and Gaza","Brunei Darussalam","Timor-Leste","Viet Nam","Taiwan","Afghanistan","Bahrain","Bangladesh","Bhutan","Brunei","Myanmar","Cambodia","China","East Timor","India","Indonesia","Iran","Iraq","Israel","Japan","Jordan","Kazakhstan","North Korea","South Korea","Kuwait","Kyrgyz Republic","Kyrgyzstan","Lao","Lebanon","Malaysia","Maldives","Mongolia","Nepal","Oman","Pakistan","Philippines","Qatar","Russia","Saudi Arabia","Singapore","Sri Lanka","Syria","Tajikistan","Thailand","Turkey","Turkmenistan","United Arab Emirates","Uzbekistan","Vietnam","Yemen"],
  "Europe": ["Slovak republic","USSR","West Germany","East Germany","Gibraltar","Macedonia [FYROM]","Slovakia","Czech Rep.","Yugoslavia","Czechoslovakia","Serbia and Montenegro","Channel Islands",u"Åland","Greenland","Macedonia, FYR","Russian Federation","Serbia and Montenegro","Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Latvia","Liechtenstein","Lithuania","Luxembourg","Macedonia","Malta","Moldova","Monaco","Montenegro","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovak Republic","Slovenia","Spain","Sweden","Switzerland","Ukraine","United Kingdom","Vatican City"],
  "North America": ["Aruba","Virgin Islands (U.S.)","Netherlands Antilles","Guadeloupe","Martinique","St. Kitts and Nevis","Bermuda","British Virgin Islands","Cayman Islands","St. Vincent and the Grenadines","Turks and Caicos Islands","St. Lucia","Dominican Rep.","Anguilla","Montserrat","Puerto Rico","Antigua and Barbuda","Bahamas","Barbados","Belize","Canada","Costa Rica","Cuba","Dominica","Dominican Republic","El Salvador","Grenada","Guatemala","Haiti","Honduras","Jamaica","Mexico","Nicaragua","Panama","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Trinidad and Tobago","United States of America","United States"],
  "Oceania": ["Guam","New Caledonia","Northern Mariana Islands","Tokelau","Cook Islands","Wallis et Futuna","Micronesia, Fed. Sts.","American Samoa","Cook Is","French Polynesia","Niue","Australia","Fiji","Kiribati","Marshall Islands","Micronesia","Nauru","New Zealand","Palau","Papua New Guinea","Samoa","Solomon Islands","Tonga","Tuvalu","Vanuatu"],
  "South America": ["French Guiana","Venezuela (Bolivarian Republic of)","Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela"]
}

continents_with_african_regions = {
  "North, Central, and Western Africa": ["Cote D'Ivoire","Mayotte",u"São Tomé and Príncipe","Congo [Republic]","Congo [DRC]",u"Côte d'Ivoire","Western Sahara","Central African Rep.","Congo, Dem. Rep.","Algeria","Benin","Burkina","Burkina Faso","Burundi","Cameroon","Cape Verde","Central African Republic","Chad","Comoros","Congo","Congo, Rep.","Djibouti","Egypt","Equatorial Guinea","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Cote d'Ivoire","Liberia","Libya","Mali","Mauritania","Morocco","Namibia","Niger","Nigeria","Sao Tome and Principe","Senegal","Seychelles","Sierra Leone","Somalia","Sudan","Togo","Tunisia"],
  "East and Southern Africa": ["Eritrea","Ethiopia","South Sudan","Uganda","Kenya","Tanzania","Angola","Zambia","Malawi","Mozambique","Rwanda","Zimbabwe","Botswana","South Africa","Swaziland","Lesotho","Reunion","Madagascar","Mauritius"],
  "Asia": ["Myanmar [Burma]","Democratic Republic of Timor-Leste","Macao, China","Laos","Libyan Arab Jamahiriya","Korea, Dem. Rep.","Yemen, Rep.","Hong Kong, China","Syrian Arab Republic","Korea, Rep.","West Bank and Gaza","Brunei Darussalam","Timor-Leste","Viet Nam","Taiwan","Afghanistan","Bahrain","Bangladesh","Bhutan","Brunei","Myanmar","Cambodia","China","East Timor","India","Indonesia","Iran","Iraq","Israel","Japan","Jordan","Kazakhstan","North Korea","South Korea","Kuwait","Kyrgyz Republic","Kyrgyzstan","Lao","Lebanon","Malaysia","Maldives","Mongolia","Nepal","Oman","Pakistan","Philippines","Qatar","Russia","Saudi Arabia","Singapore","Sri Lanka","Syria","Tajikistan","Thailand","Turkey","Turkmenistan","United Arab Emirates","Uzbekistan","Vietnam","Yemen"],
  "Europe": ["Slovak republic","USSR","West Germany","East Germany","Gibraltar","Macedonia [FYROM]","Slovakia","Czech Rep.","Yugoslavia","Czechoslovakia","Serbia and Montenegro","Channel Islands",u"Åland","Greenland","Macedonia, FYR","Russian Federation","Serbia and Montenegro","Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Latvia","Liechtenstein","Lithuania","Luxembourg","Macedonia","Malta","Moldova","Monaco","Montenegro","Netherlands","Norway","Poland","Portugal","Romania","San Marino","Serbia","Slovak Republic","Slovenia","Spain","Sweden","Switzerland","Ukraine","United Kingdom","Vatican City"],
  "North America": ["Aruba","Virgin Islands (U.S.)","Netherlands Antilles","Guadeloupe","Martinique","St. Kitts and Nevis","Bermuda","British Virgin Islands","Cayman Islands","St. Vincent and the Grenadines","Turks and Caicos Islands","St. Lucia","Dominican Rep.","Anguilla","Montserrat","Puerto Rico","Antigua and Barbuda","Bahamas","Barbados","Belize","Canada","Costa Rica","Cuba","Dominica","Dominican Republic","El Salvador","Grenada","Guatemala","Haiti","Honduras","Jamaica","Mexico","Nicaragua","Panama","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Trinidad and Tobago","United States of America","United States"],
  "Oceania": ["Guam","New Caledonia","Northern Mariana Islands","Tokelau","Cook Islands","Wallis et Futuna","Micronesia, Fed. Sts.","American Samoa","Cook Is","French Polynesia","Niue","Australia","Fiji","Kiribati","Marshall Islands","Micronesia","Nauru","New Zealand","Palau","Papua New Guinea","Samoa","Solomon Islands","Tonga","Tuvalu","Vanuatu"],
  "South America": ["French Guiana","Venezuela (Bolivarian Republic of)","Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela"]
}

def getRegion(countryName):
  for region in continents_with_african_regions:
    if countryName in continents_with_african_regions[region]:
      return region
  print "MISMATCH", countryName

reinitializeFeatureVectors = True
# This next line creates the feature vectors from scratch with imputation.
# To get rid of imputation, pass in False as the first param.  
# To use latitude and longitude, pass in True as the second param.
# To exclude HIV, pass in False as third param.
feature_vectors = {}
if reinitializeFeatureVectors:
  create_feature_vectors.initialize_vectors()
  pickle.dump(feature_vectors, open('featureVectors.p', 'wb'))

# Options for loading from file: featureVectors.p, imputed_feature_vectors.p, 
# and imputed_feature_vectors_no_hiv.p.
with open('imputed_feature_vectors.p') as data_file:    
  feature_vectors = pickle.load(data_file)

feature_vectors_arr = []
hiv_arr = []

for key, val in feature_vectors.iteritems():
  features, hiv = val
  if hiv is not None:
    feature_vectors_arr.append(features)
    hiv_arr.append(hiv)

# Prepare partitions
bayes_partition = partition = int(len(feature_vectors_arr) * .7)
training_arr = feature_vectors_arr[:partition]
test_arr = feature_vectors_arr[partition:]
hiv_training_arr = hiv_arr[:partition]
hiv_test_arr = hiv_arr[partition:]

print "%d elems in training_arr out of %d total" % (len(training_arr), len(feature_vectors_arr))

vertexMap = {}

condensed_feature_vectors_by_region = {}
mainFeatures = ['indicator total health expenditure perc of GDP.xlsx', 'Indicator_BMI male ASM.xlsx', 'indicator food_consumption.xlsx', 'indicator_estimated incidence infectious tb per 100000.xlsx', 'indicator life_expectancy_at_birth.xlsx', 'indicator gapminder infant_mortality.xlsx']
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
  region = getRegion(sample['Country'])
  if not region:
    continue
  if region not in condensed_feature_vectors_by_region:
    condensed_feature_vectors_by_region[region] = []
  condensed_feature_vectors_by_region[region].append(newSample)

vertexAverages = {}
for vertex, samples in vertexMap.iteritems():
  numerator = 0
  for sample in samples:
    numerator += sample[vertex]
  vertexAverages[vertex] = numerator/len(samples)

for region in condensed_feature_vectors_by_region:
  for sample in condensed_feature_vectors_by_region[region]:
    for vertex in vertices:
      if vertex not in sample.keys():
        # print "Used avg for %s" % vertex
        sample[vertex] = vertexAverages[vertex]

# Instantiate and run learner 
learner = PGMLearner()
bayesian_networks_by_region = {}
for region in condensed_feature_vectors_by_region:
  print region
  bayesian_networks_by_region[region] = learner.lg_estimatebn(condensed_feature_vectors_by_region[region])
  print json.dumps(bayesian_networks_by_region[region].Vdata, indent=2)
  print json.dumps(bayesian_networks_by_region[region].E, indent=2)

# Evaluation
test_arrs_by_region = {}
hiv_test_arrs_by_region = {}
for i, sample in enumerate(test_arr):
  region = getRegion(sample['Country'])
  if not region:  
    continue
  if region not in test_arrs_by_region:   
    test_arrs_by_region[region] = []
    hiv_test_arrs_by_region[region] = []
  test_arrs_by_region[region].append(sample)
  hiv_test_arrs_by_region[region].append(hiv_test_arr[i])


for region in bayesian_networks_by_region:
  print region
  print np.std(hiv_test_arrs_by_region[region])
  predictions = []
  for sample in test_arrs_by_region[region]:
    predictionSamples = bayesian_networks_by_region[region].randomsample(500, sample)
    hivPredictionSamples = [pSample['HIV'] for pSample in predictionSamples]
    predictions.append(np.mean(hivPredictionSamples))

  result_pairs = [(predictions[i], hiv_test_arrs_by_region[region][i]) for i in range(len(hiv_test_arrs_by_region[region]))]

  predictions_std = preprocessing.scale(predictions)

  # With standardized predictions, SD = 1, so we're just getting how many SDs off we are
  # test_sd = np.std(hiv_test)
  # print "Sd is %f" % test_sd

  # Without normalizing predictions, we need to use hiv_test_arr because that has the unnormalized HIV rates.
  # Now the error units are HIV rate percentage points.
  print "Mean absolute error is %f" % mean_absolute_error(hiv_test_arrs_by_region[region], predictions)
  print "Median absolute error is %f" % median_absolute_error(hiv_test_arrs_by_region[region], predictions) 
  print "Mean squared error is %f" % mean_squared_error(hiv_test_arrs_by_region[region], predictions) 

