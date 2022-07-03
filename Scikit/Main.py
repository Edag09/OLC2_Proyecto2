from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle

#cargar datos en data set
iris = datasets.load_iris()

X = iris.data
Y = iris.target

