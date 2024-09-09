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
from nd_hand_fashion_valuation.preprocessor import api_preprocessor

app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#http://127.0.0.1:8000/predict?product_category=Women%20Clothing&product_season=Autumn%20%2F%20Winter&product_condition=Never%20worn&brand_name=Barbara%20Bui&seller_badge=Common&seller_products_sold=2&product_material=Wool&shipping_days=3-5%20days&product_color=Navy&product_gender=Women&product_description=small

app.state.model = pickle.load(open("models/model.pkl","rb"))

@app.get("/predict")
def predict(
        product_category: str,      # Women Clothing
        product_season: str,        # Autumn / Winter
        product_condition: str,     # Never worn
        brand_name: str,            # Miu Miu
        seller_badge: str,          # Common
        seller_products_sold: float, # 3.0
        product_material: str,         # Natural Fibers
        shipping_days: str,        # 1.5
        product_color: str,            # Neutrals
        product_gender: str,           # 1
        product_description: str
    ):

    #X_pred = pd.DataFrame(locals(), index=[0]) #locals() get all fn arg as dict

    X = [[product_category,
        product_season,
        product_condition,
        brand_name,
        seller_badge,
        seller_products_sold,
        product_material,
        shipping_days,
        product_color,
        product_gender,
        product_description]]

    X_pred = pd.DataFrame(data=X, columns=['product_category',
                                        'product_season',
                                        'product_condition',
                                        'brand_name',
                                        'seller_badge',
                                        'seller_products_sold',
                                        'material_group',
                                        'shipping_days',
                                        'color_group',
                                        'gender_binary',
                                        'cleaned_description'])


    model = app.state.model
    assert model is not None
    print("---- Model has been loaded ----")



    X_preprocess = api_preprocessor(X_pred)

    pipeline = pickle.load(open("models/pipeline.pkl","rb"))
    print("----- Pipeline pickle has been loaded ------")

    X_processed = pipeline.transform(X_preprocess)
    print("----- X_pred has been transformed  ------")

    y_pred = model.predict(X_processed)

    y_pred = np.exp(y_pred)

    return dict(price = round(float(y_pred),2))



@app.get("/")
def root():
    return {'greeting': 'Hello'}
