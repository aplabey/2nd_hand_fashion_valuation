# install below lib before running this file
# pip install uvicorn
# pip install fastapi

import pandas as pd
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pickle
#import sys

#sys.path.append("nd_hand_fashion_valuation")
#from nd_hand_fashion_valuation.preprocessor import preprocess_features

app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#http://127.0.0.1:8000/predict?product_category=Women%20Clothing&product_season=Autumn%20%2F%20Winter&product_condition=Never%20worn&brand_name=Miu%20Miu&seller_badge=Common&seller_products_sold=3.0&material_group=Natural%20Fibers&shipping_days=1.5&color_group=Neutrals&gender_binary=1

app.state.model = pickle.load(open("models/model.pkl","rb"))

@app.get("/predict")
def predict(
        product_category: str,      # Women Clothing
        product_season: str,        # Autumn / Winter
        product_condition: str,     # Never worn
        brand_name: str,            # Miu Miu
        seller_badge: str,          # Common
        seller_products_sold: float, # 3.0
        material_group: str,         # Natural Fibers
        shipping_days: float,        # 1.5
        color_group: str,            # Neutrals
        gender_binary: int           # 1
    ):

    #X_pred = pd.DataFrame(locals(), index=[0]) #locals() get all fn arg as dict

    X = [[product_category,
        product_season,
        product_condition,
        brand_name,
        seller_badge,
        seller_products_sold,
        material_group,
        shipping_days,
        color_group,
        gender_binary]]

    X_pred = pd.DataFrame(data=X, columns=['product_category',
                                        'product_season',
                                        'product_condition',
                                        'brand_name',
                                        'seller_badge',
                                        'seller_products_sold',
                                        'material_group',
                                        'shipping_days',
                                        'color_group',
                                        'gender_binary'])


    model = app.state.model
    assert model is not None
    print("---- Model has been loaded ----")

    pipeline = pickle.load(open("models/pipeline.pkl","rb"))
    print("----- Pipeline pickle has been loaded ------")

    X_processed = pipeline.transform(X_pred)
    print("----- X_pred has been transformed  ------")

    y_pred = model.predict(X_processed)

    y_pred = np.exp(y_pred)

    return dict(price = round(float(y_pred),2))



@app.get("/")
def root():
    return {'greeting': 'Hello'}
