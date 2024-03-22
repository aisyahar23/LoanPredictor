#AISYAH BT ABDUL RAZAK
#The project about the loan prediction
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Union
import time

#uses base model for json dictionary
class LoanApplication(BaseModel):
    Loan_ID: Union[str, None]
    Gender: Union[str, None]
    Married: Union[str, None]
    Dependents: Union[str, None]
    Education: Union[str, None]
    Self_Employed: Union[str, None]
    ApplicantIncome: Union[int, None]
    CoapplicantIncome: Union[float, None]
    LoanAmount: Union[float, None]
    Loan_Amount_Term: Union[float, None]
    Credit_History: Union[float, None]
    Property_Area: Union[str, None]

#initialize the app
app = FastAPI()

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Default route uses 8000 route
@app.get("/")
async def root():
    return {"Hello": "World"}

# Predict route
@app.post("/predict")
async def predict(request: LoanApplication):
    #start to measure the time taken
    startTime = time.time() 

    data = request.model_dump()
    df = pd.DataFrame([data])
    
    # Handling missing values by replacing with appropiate data
    df['Credit_History'].fillna(1, inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['Dependents'].fillna("0", inplace=True)
    df['Gender'].fillna("Male", inplace=True)

    #remove data null value
    df = df.dropna()

    #input the wrong data
    #for example it suppose to be string but input null
    #will causes error
    if df.empty:
        return {"status": "failed", "error_code": 1, "prediction": {"label": None, "value": None}, "time_taken": "0ms"}

    # processing the data 
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Married'] = df['Married'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
    df['Education'] = df['Education'].apply(lambda x: 1 if x == 'Graduate' else 0)
    df['Self_Employed'] = df['Self_Employed'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Rural': 0, 'Semiurban': 1})

    # make  the prediction
    input_data = df.drop(columns=['Loan_ID'])
    prediction = model.predict(input_data)
    predictionClass = int(prediction[0])  # Convert to Python int
    predictionClassName = "Approve" if predictionClass == 1 else "Reject"

    endTime = time.time()  # End timing
    finalTime = (endTime - startTime) * 1000  # Convert to milliseconds

    return {
        "status": "success",
        "error_code": 0,
        "prediction": {
            "label": predictionClass,
            "value": predictionClassName
        },
        "time_taken": f"{finalTime:.0f}ms"  # Use the processing time obtained within the function
    }
