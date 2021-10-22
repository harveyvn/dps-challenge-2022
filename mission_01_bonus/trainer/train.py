# This will be replaced with your bucket name after running the `sed` command in the tutorial
BUCKET = "gs://dps-challenge-329218-bucket"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

print(tf.__version__)

def load_data():
    ''' 
    load original train data and test data from csv files
    '''
    X_full = pd.read_csv("https://raw.githubusercontent.com/harveyvn/dataset/main/home-data-for-ml-course/train.csv")
    X_test_full = pd.read_csv("https://raw.githubusercontent.com/harveyvn/dataset/main/home-data-for-ml-course/test.csv")
    return X_full, X_test_full
    
def preprocess_missing_value_columns(df):
    ''' 
    handle features having missing values from data frames
    '''
    for col in ["MasVnrType", "MSSubClass", "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]:  
        df[col].fillna("None", inplace=True)
        
    for col in ["MasVnrArea", "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF","TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]:
        df[col] = df[col].fillna(0)
        
    for col in ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean())
    
    df.drop(["Utilities"], axis=1, inplace=True)

    return df

def preprocess_date_month_columns(df):
    ''' 
    drop date month features
    '''
    for col in ["MSSubClass", "OverallCond", "YrSold", "MoSold"]:
      df.drop([col], axis=1, inplace=True)
    return df

def preprocess_categorical_values_with_label_encoder(df):
    from sklearn.preprocessing import LabelEncoder
    column = ["FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "ExterQual", 
              "ExterCond","HeatingQC", "PoolQC", "KitchenQual", "BsmtFinType1", "BsmtFinType2", 
              "Functional", "Fence", "BsmtExposure", "GarageFinish", "LandSlope", "LotShape", 
              "PavedDrive", "Street", "Alley", "CentralAir", 
#               "MSSubClass", "OverallCond", "YrSold", "MoSold", "YrMoSold"
             ]
    
    for col in column:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(list(df[col].values))
    
    return df

def normalize_data(df_train, df_test):
    from sklearn.preprocessing import MinMaxScaler
    ''' 
    normalize data
    '''
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train)
    X_test = scaler.transform(df_test)

    X_train = pd.DataFrame(X_train, index=df_train.index, columns=df_train.columns)
    X_test = pd.DataFrame(X_test, index=df_test.index, columns=df_test.columns)

    return X_train, df_test
    
def preprocess(df_train, df_test):
    ''' 
    bundle all functions to preprocess data
    '''
    full_data = pd.concat([df_train.iloc[:, 1:-1], df_test.iloc[:, 1:]], ignore_index=True, sort=False)

    full_data = preprocess_missing_value_columns(full_data)
    full_data = preprocess_date_month_columns(full_data)
    full_data = preprocess_categorical_values_with_label_encoder(full_data)
    full_data = pd.get_dummies(full_data)
    
    # 4. split all data into train and test
    train_num = df_train.shape[0]
    X_train = full_data[:train_num]
    X_test = full_data[train_num:]
    
    return normalize_data(X_train, X_test)
#     return X_train, X_test

def build_model(X_train):
    # Build the Keras Sequential model
    model = keras.Sequential()
    model.add(Dense(416, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(192, activation='relu'))
    model.add(Dense(320, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(448, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Configure the training procedure using the Keras Model.compile method
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_absolute_error"
    )
    
    return model

def prepare_data():
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error
    
    X_full, X_test_full = load_data()
    X = X_full.copy()
    y = X.SalePrice
    X.to_csv("data_train.csv", sep='\t', encoding='utf-8')
    X.drop(['SalePrice'], axis=1, inplace=True)

    X_test = X_test_full.copy()
    X_test.to_csv("data_test.csv", sep='\t', encoding='utf-8')

    X, X_test = preprocess(X, X_test)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    
    
    return X_train, X_valid, y_train, y_valid



X_train, X_valid, y_train, y_valid = prepare_data()
model = build_model(X_train)

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_split=0.2, callbacks=[early_stop])


# Export model and save to GCS
model.save(BUCKET + '/saleprice_custom/model')