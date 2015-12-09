from openpyxl import *
from collections import defaultdict
import os

# makes feature vectors from all .xlsx files in the working directory
def create_feature_vectors():
	feature_vectors = {}
	target_values = defaultdict(lambda: None)
	start_year = 1990

	target_wb = load_workbook(filename = "indicator hiv estimated prevalence% 15-49.xlsx")
	target_ws = target_wb['Data']

	# store target values
	for row in target_ws.iter_rows(row_offset=1):
		country = row[0].value
		if country == None:
			break
		for cell in row:
			if cell.column != 'A':
				year = int(target_ws[cell.column + '1'].value)
				if year >= start_year:
					target_values[(country, year)] = cell.value


	# build feature vectors
	for filename in os.listdir(os.getcwd()):
		if filename.endswith(".xlsx") and not filename.startswith("~$") and not filename == "indicator hiv estimated prevalence% 15-49.xlsx": 
			print filename
			wb = load_workbook(filename = filename)

			feature = filename

			ws = wb['Data']

			for row in ws.iter_rows(row_offset=1):
				country = row[0].value
				if country == None:
					break
				for cell in row:
					if cell.value != None and cell.column != 'A':
						year = int(ws[cell.column + '1'].value)
						if year >= start_year:
							if (country, year) not in feature_vectors.keys():
								target = target_values[(country, year)]
								feature_vectors[(country, year)] = ({'Country': country, 'Year': year}, target)
							feature_vectors[(country, year)][0][feature] = cell.value
			
			for vector in feature_vectors.values():
				print vector

	return feature_vectors
