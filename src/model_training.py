# Load libraries
import pickle
from numpy import dtype
from pandas import read_csv
from pandas.core.frame import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os


# Load dataset
currrentPath = os.path.dirname(os.path.abspath(__file__))

parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))

url = parentPath+'/data/PdiabetesDATASET.csv'
dataset = read_csv(url)


# dataset manipulation
## Diabetes column values ["neg","pos"] are changed into 0s and 1s
dataset.loc[dataset["diabetes"] == "neg", "targetADD"] = 0 #np.random.choice(346)\
dataset.loc[dataset["diabetes"] == "pos", "targetADD"] = 1 #np.random.choice(346)
#target column is created as a numerical value as seen in the website example
dataset = dataset.assign(target = lambda x: (x["pedigree"]*11+x['targetADD']))


## Columns are dropped (insuling is a binary 0/1 column)
dataset = dataset.drop("targetADD",axis=1)
dataset = dataset.drop("insulin",axis=1)

diabetesdfFULL = dataset
diabetesdfFULL = diabetesdfFULL.drop("diabetes",axis=1)

data2NOTARGET = dataset
data2NOTARGET = data2NOTARGET.drop(["diabetes"],axis = 1)
data2NOTARGET = data2NOTARGET.drop(["target"],axis = 1)

# Split-out validation dataset
X_train, X_validation, Y_train, Y_validation = train_test_split(data2NOTARGET, diabetesdfFULL["target"], test_size=0.2, random_state=0)
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Spot Check Algorithms
# print("checkpoint")
model = LinearRegression()
# print("checkpoint2")
model.fit(X_train, Y_train)

# Save model

filename = parentPath+'/modelDiabetesModel/modelDiabetesModel'

outfile = open(filename,'wb')

pickle.dump(model, outfile)

outfile.close()






