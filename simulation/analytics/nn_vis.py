from sklearn.neural_network import MLPClassifier
import pickle
from pprint import pprint
model1 = MLPClassifier()
# model1.co
model = pickle.load(open("models/mlp_classifier.model", "rb"))
pprint(model.coefs_)