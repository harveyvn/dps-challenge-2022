# This will be replaced with your bucket name after running the `sed` command in the tutorial
BUCKET = "gs://dps-challenge-329218-bucket"

import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])

install("category_encoders")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
# Turn off the warning altogether
pd.set_option('mode.chained_assignment',None)

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
names = ["mpg", "cylinders", "displacement", "horsepower", "weight","acceleration", "model year", "origin", "car name"]
widths = [7, 4, 10, 10, 11, 7, 4, 4, 30]

# Get the data
X_full = pd.read_fwf(url, names=names, widths=widths, na_values=['?'])
X = X_full.copy()

# A dictionary of companies getting from a feature "car name"
brands_dict = {
    "amc": "AMC",
    "audi": "Audi",
    "bmw": "Bmw",
    "buick": "Buick",
    "cadillac": "Cadillac",
    "capri": "Capri",
    "chevroelt": "Chevrolet",
    "chevrolet": "Chevrolet",
    "chevy": "Chevrolet",
    "chrysler": "Chrysler",
    "datsun": "Datsun",
    "dodge": "Dodge",
    "fiat": "Fiat",
    "ford": "Ford",
    "hi": "IH",
    "honda": "Honda",
    "maxda": "Mazda",
    "mazda": "Mazda",
    "mercedes": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "mercury": "Mercury",
    "nissan": "Nissan",
    "oldsmobile": "Oldsmobile",
    "opel": "Opel",
    "peugeot": "Peugeot",
    "plymouth": "Plymouth",
    "pontiac": "Pontiac",
    "renault": "Renault",
    "saab": "Saab",
    "subaru": "Subaru",
    "toyota": "Toyota",
    "toyouta": "Toyota",
    "triumph": "Triumph",
    "vokswagen": "Volkswagen",
    "volkswagen": "Volkswagen",
    "volvo": "Volvo",
    "vw": "Volkswagen"
}


# Create a new feature named Company
X["company"] = [brands_dict[X["car name"][i].replace('"', '').split()[0]] for i in range(len(X["car name"]))]

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=["mpg"], inplace=True)
y = X["mpg"]
X.drop(["mpg"], axis=1, inplace=True)

# Imputation
hp_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputed_hp_train = hp_imputer.fit(X[["horsepower"]])

# Put them back to X_train and X_valid dataframe
X["hp"] = imputed_hp_train.transform(X[["horsepower"]]).ravel()


# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[["origin"]]))

# One-hot encoding removed index; put it back
OH_cols.index = X.index

# Add one-hot encoded columns to numerical features
X = pd.concat([X, OH_cols], axis=1)

# Apply Target Encoder to feature company
encoder = TargetEncoder()
encoded_comp = encoder.fit(X[["company"]], y)

# Put them back to X_train and X_valid dataframe
X["company_encode"] = encoded_comp.transform(X[["company"]])


# Select features we will use to train the model
features = ["cylinders", "displacement", "weight", "acceleration", "model year", "hp", 0, 1, 2, "company_encode"]


def dnn_regression_multi(columns):
    # Build the Keras Sequential model
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(features)]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    # Configure the training procedure using the Keras Model.compile method
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="mean_absolute_error"
    )

    return model

    # Execute the training for 100 epochs


    return model, history


model = dnn_regression_multi(features)

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X[features], y, epochs=EPOCHS, verbose=1, validation_split=0.2, callbacks=[early_stop])

# Export model and save to GCS
model.save(BUCKET + '/mpg-custom/model')