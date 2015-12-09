
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from openpyxl import *
from collections import defaultdict
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import os, pickle, operator


with open('featureVectors.p') as data_file:    
    feature_vectors = pickle.load(data_file)

def impute(target_name):
  print "Imputing for %s" % target_name
  feature_vectors_arr = []
  val_arr = []
  real_inputs_arr = []

  for key, val in feature_vectors.iteritems():
    features, hiv = val
    if hiv is not None: features["HIV rate"] = hiv
    if target_name in features:
      cur_val = features[target_name]
      del features[target_name]
      feature_vectors_arr.append(features)
      val_arr.append(cur_val)
    else:
      real_inputs_arr.append(features)

  all_feature_vectors = feature_vectors_arr + real_inputs_arr
  vec = DictVectorizer()
  all_data = vec.fit_transform(all_feature_vectors).toarray()
  feature_names = vec.get_feature_names()


  data = all_data[:len(feature_vectors_arr)]
  real_data = all_data[len(feature_vectors_arr):]

  partition = int(len(data) * .7)
  training_data = preprocessing.scale(data[:partition])
  test_data = preprocessing.scale(data[partition:])
  target_training = preprocessing.scale(val_arr[:partition])
  target_test = preprocessing.scale(val_arr[partition:])
  real_input_data = preprocessing.scale(real_data)

  # # print training_data[0]
  # # print target_training
  print "%d in training and %d in test" % (len(target_training), len(target_test))
  print "Feature test len: %d  target test len: %d" % (len(test_data), len(target_test))

  # fit the unweighted model
  clf = linear_model.SGDRegressor()
  clf.fit(training_data, target_training)
  predictions = clf.predict(test_data)

  # Check error rates
  test_sd = np.std(target_test)
  print "Sd is %f" % test_sd
  print mean_absolute_error(target_test, predictions) * test_sd
  print median_absolute_error(target_test, predictions) * test_sd
  print mean_squared_error(target_test, predictions) * test_sd
  print "Mean absolute error is %f" % mean_absolute_error(target_test, predictions) * test_sd
  print "Median absolute error is %f" % median_absolute_error(target_test, predictions) * test_sd
  print "Mean squared error is %f" % mean_squared_error(target_test, predictions) * test_sd

  print "----AND FINALLY, THE IMPUTATION---"
  imputations = clf.predict(real_input_data)
  output_sd = np.std(val_arr[:partition])
  output_mean = np.mean(val_arr[:partition])
  scaled_imputations = [val * output_sd + output_mean for val in imputations]
  imputed_vals = {}
  for i, features in enumerate(real_inputs_arr):
    # print "Country: %s, Year: %d, %s: %f"  % (features['Country'], features['Year'], target_name, scaled_imputations[i])
    imputed_vals[(features['Country'], features['Year'], target_name)] = scaled_imputations[i]
  return imputed_vals


all_imputed_vals = {}
health_spending = impute('indicator total health expenditure perc of GDP.xlsx')
infant_mortality = impute('indicator gapminder infant_mortality.xlsx')
male_bmi = impute('Indicator_BMI male ASM.xlsx')
tb_incidence = impute('indicator_estimated incidence infectious tb per 100000.xlsx')
food_consumption = impute('indicator food_consumption.xlsx')
life_expectancy = impute('indicator life_expectancy_at_birth.xlsx')
hiv = impute('HIV rate')

all_imputed_vals.update(health_spending)
all_imputed_vals.update(infant_mortality)
all_imputed_vals.update(male_bmi)
all_imputed_vals.update(tb_incidence)
all_imputed_vals.update(food_consumption)
all_imputed_vals.update(life_expectancy)
all_imputed_vals.update(hiv)


# Now all the imputing is done, just add back in to feature vector and save in pickle

with open('featureVectors.p') as data_file:    
    feature_vectors_fresh = pickle.load(data_file)

imputed_feature_vectors = {}
for key, val in feature_vectors_fresh.iteritems():
  features, hiv = val
  all_features = {}
  all_features.update(features)
  country = all_features['Country']
  year = all_features['Year']
  imputed = ['indicator total health expenditure perc of GDP.xlsx', 'indicator gapminder infant_mortality.xlsx', 'Indicator_BMI male ASM.xlsx', 'indicator_estimated incidence infectious tb per 100000.xlsx', 'indicator food_consumption.xlsx', 'indicator life_expectancy_at_birth.xlsx']
  for name in imputed:
    if (country, year, name) in all_imputed_vals:
      all_features[name] = all_imputed_vals[(country, year, name)]
    if name not in all_features: 
      print "Missing %s for %s in %d" % (name, country, year)

  if hiv is None: 
    hiv = all_imputed_vals[(country, year, 'HIV rate')]

  imputed_feature_vectors[key] = (all_features, hiv)


# print imputed_feature_vectors

pickle.dump(imputed_feature_vectors, open('imputed_feature_vectors.p', 'wb'))



  


# print all_imputed_vals
