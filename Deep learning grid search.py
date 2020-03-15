import sys
import pandas as pd
import keras
import numpy as np
import sklearn
import theano
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.layers import Dropout
from sklearn.metrics import classification_report, accuracy_score
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10000)

# import the uci pima indians diabetes dataset
url = "C:\\Users\\hshah\\Desktop" \
      "\\building machine learning project with python" \
      "\\Diabetes onset detection using deep learning grid search" \
      "\\pima-indians-diabetes-database" \
      "\\diabetes.csv"

names = ["n_preganant","glucose_concentration",
         "blood_pressure mm Hg","skin_thickness (mm)",
         "serum_insulin (mu U/ml)","BMI",
         "pedigree_function","age",
         "class"]

df = pd.read_csv(url, names = names)

# decribe the dataset
# print(df.describe())

# print(df[df["glucose_concentration"] == 0])

# preprocess the data, mark zero values as NaN and drop them
columns = ["glucose_concentration",
         "blood_pressure mm Hg","skin_thickness (mm)",
         "serum_insulin (mu U/ml)","BMI"]

for col in columns:
      df[col].replace(0,np.NAN, inplace=True)

# print(df.describe())

# drop rows with missing values
df.dropna(inplace=True)

# summarize the number of rows and columns in df
# print(df.describe())

# convert dataframe to numpy array
dataset = df.values
# print(dataset)

#split data into input(X) and output(y)
X = dataset[:,0:8]
y = dataset[:,8].astype(int)

# print(X.shape)
# print(y.shape)
# print(y[:5])

# normalize the data using sklearn StandardScalar

scaler = StandardScaler().fit(X)
# print(scaler)

# transfer and display the training data
X_standardized = scaler.transform(X)

data = pd.DataFrame(X_standardized)
# print(data.describe())

# define a random seed
seed = 6
np.random.seed(seed)

# start defining the model

def create_model(learn_rate, dropout_rate,activation,init,neuron1,neuron2):
    # create model
    model = Sequential()
    model.add(Dense(neuron1,input_dim=8, kernel_initializer=init,activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neuron2, input_dim=neuron1,kernel_initializer=init,activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation="sigmoid"))

    # compile the model
    adam = Adam(lr=learn_rate)
    model.compile(loss = "binary_crossentropy",optimizer=adam,metrics=["accuracy"])
    return model

# model = create_model()
# print(model.summary())

# create the model
model = KerasClassifier(build_fn=create_model,epochs=10,batch_size=20, verbose = 0)

# define the grid search parameters
# batch_size = [10,20,40]
# epochs = [10,50,100]

# define the grid search parameters
learn_rate = [0.1]
dropout_rate = [0.1]
activation = ["relu"]
init = ["uniform"]
neuron1 = [4,8,16]
neuron2 = [2,4,8]

# make a dictionary of the grid parameters
param_grid = dict(learn_rate=learn_rate,dropout_rate=dropout_rate,
                  activation=activation,init=init,neuron1=neuron1,neuron2=neuron2)

# build and fit the GridSearchCV
grid = GridSearchCV\
    (estimator=model,
     param_grid=param_grid,
     cv=KFold(random_state=seed),
     refit=True,
     verbose = 10)
grid_results = grid.fit(X_standardized,y)

# summarize the results
print("Best: {0}, "
      "using {1}"
      .format(grid_results.best_score_,
              grid_results.best_params_))

means = grid_results.cv_results_["mean_test_score"]
stds = grid_results.cv_results_["std_test_score"]
params = grid_results.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    print("{0} ({1}) with: {2}".format(mean,stdev,param))

# generate predictions with optima hyperparameters

y_pred = grid.predict(X_standardized)

print(y_pred.shape)
print(y_pred[:5])

# generate a classification report

print(accuracy_score(y,y_pred))
print(classification_report(y,y_pred  ))

# example datapoint
example = df.iloc[99]
print(example)

# make a prediction using optimized neural network
prediction = grid.predict(X_standardized[99].reshape(1, -1))
print(prediction)