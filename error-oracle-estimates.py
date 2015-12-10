from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import numpy
import csv, re

#Source for predictions: http://www.popcouncil.org/uploads/pdfs/wp/pgy/009.pdf
#Source for actual: http://aidsinfo.unaids.org/#
#2010 and 2014 projections by continent as array
projected = [5,4.6,0.5,0.6,0.19,0.15,0.75,0.75]
actual = [7.8,7.4,0.3,0.3,0.3,0.3,0.8,0.8]

print "Mean absolute error is %f" % mean_absolute_error(projected, actual)
print "Median absolute error is %f" % median_absolute_error(projected, actual) 
print "Mean squared error is %f" % mean_squared_error(projected, actual) 

def avg_margin_of_error_range():
  ranges = []
  with open('People living with HIV_HIV Prevalence among adults_by_country (15-49).csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      for cell in row:
        # print cell
        match = re.search('\[\d.\d - \d.\d]', cell)
        if match:
          rangeString = match.group(0)
          # print rangeString
          rangeVal = float(rangeString[7:10]) - float(rangeString[1:4])
          ranges.append(rangeVal)

  # Divide by 2 because margin of error is 1/2 of range
  return numpy.mean(ranges) / 2

# print avg_margin_of_error_range()

# Result = 0.314407744875


