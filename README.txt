Commands to run:
python sgd.py
- Parses feature vectors from data, which are stored in .xlsx files in the same folder, or loads the vectors from a .p file
- Learns a linear model using linear regression and stochastic gradient descent
- Prints the test error

python bayesian-only.py
- Parses or loads feature vectors, like above
- Learns a Bayesian network from the data, with variables specified in the code
- Prints the Bayesian network graph structure and parameters, along with the test error

python bayesian-continent.py
- Same as bayesian-only, but it makes regional (primarily continent-wide) Bayesian networks.

Other files:
create_feature_vectors.py: Parses feature vectors from the data. Used by sgd.py, bayesian-only.py, and bayesian-continent.py.
impute.py: Imputes missing values using linear regression. Used by create_feature_vectors.py.