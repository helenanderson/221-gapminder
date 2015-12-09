import csv
import os, pickle, operator

 # -*- coding: utf-8 -*-

def country_to_lat_long():
  mappings = {}
  with open('countries.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      mappings[row[2]] = (float(row[0]), float(row[1]))
  # print mappings
  return mappings

def get_mappings(feature_vectors):
  mappings = country_to_lat_long()
  mappings[u'\xc5land'] = (60.2105,20.2750)
  mappings[u'S\xe3o Tom\xe9 and Pr\xedncipe'] = (0.3333,6.7333)
  mappings[u"C\xf4te d'Ivoire"] = (6.8500,-5.3000)
  # with open('imputed_feature_vectors.p') as data_file:    
  #   feature_vectors = pickle.load(data_file)
  missing = set()
  for key, val in feature_vectors.iteritems():
    features, hiv = val
    country, year = key
    if country in mappings:
      lat, longt = mappings[country]
      features['lat'] = lat
      features['long'] = longt
      features['lat-squared'] = lat ** 2
      features['long-squared'] = lat ** 2
    # else:
    #   missing.add(country)
  # print missing
  return mappings



# print country_to_lat_long()
# get_mappings()
