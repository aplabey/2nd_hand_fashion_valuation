#import streamlit as st
#import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle


app = FastAPI()





# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# test case for predict to see whether api is working
# this will be changed with load model
#http://127.0.0.1:8000/predict?day_of_week=7&time=6

#app.state.model = pickle.load(open("../model/model.pkl","rb"))

@app.get("/predict")
def predict(day_of_week, time):

    wait_prediction = int(day_of_week) * int(time)

    #model = app.state.model
    #assert model is not None

    #X_processed = preprocess_features(X_pred)

    #y_pred = model.predict(X_processed)


    return {'wait': wait_prediction}



@app.get("/")
def root():
    return {'greeting': 'Hello'}
