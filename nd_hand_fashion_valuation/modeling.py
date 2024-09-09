# install below lib before running this lib
# pip install xgboost

import pandas as pd
import numpy as np
import pickle
import sys

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


from nd_hand_fashion_valuation.preprocessor import preprocess_features, preproc_pipe


# reading raw data

df = pd.read_csv('raw_data/vestiaire.csv')

# filling the missing values and cleaning the data

df_cleaned = preprocess_features(df)

print("----- Data has been cleaned ------")

# separating target and features

y = df_cleaned['price_usd']
y_log = np.log(y)

X = df_cleaned.drop(columns=['price_usd'])

# transforming features

X_processed = preproc_pipe(X, y_log)

# obtaining processed X_train & X_test and the target y_train & y_test

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_log, test_size=0.3)

print("----- Data has been transformed and split ------")

#preparing model

model_xgb_reg = XGBRegressor(max_depth=10, n_estimators=150, learning_rate=0.2)

model_xgb_reg.fit(X_train, y_train)

score = model_xgb_reg.score(X_test, y_test)

print('The score of the model is: ', score)

#Export model as pickle file
with open("models/model.pkl", "wb") as file:
    pickle.dump(model_xgb_reg, file)
    print("----- model pickle file has been generated -----")
