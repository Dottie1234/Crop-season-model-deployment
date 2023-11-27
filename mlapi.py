from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

with open('nb_model.pkl', 'rb') as file:
    model = pickle.load(file)

encoder = joblib.load('encoder.joblib')

class FeaturesInput(BaseModel):
    temperature: float
    humidity: float
    ph: float
    water_availability: float
    crop: str
    country: str
    
@app.post('/predict')
def predict(data: FeaturesInput):
    try:
        data_dict = data.model_dump()
        new_data_dict = {key: [value] for key, value in data_dict.items()}
        new_data = pd.DataFrame(new_data_dict)
        new_data.replace({'crop':{'blackgram':0,
                 'chickpea': 1,
                 'cotton': 2,
                 'jute':3,
                 'kidneybeans':4,
                 'lentil':5,
                 'maize':6,
                 'mothbeans':7,
                 'mungbean': 8,
                 'muskmelon':9,
                 'pigeonpeas':10,
                 'rice':11,
                 'watermelon':12},
         'country':{'Kenya':0,
                   'Nigeria': 1,
                   'South Africa': 2,
                   'Sudan': 3}}, inplace=True)

        #print(data)
        #new_data.to_numpy()
        prediction = model.predict(new_data)

        return {'prediction': str(prediction)}
        #return {'message': str(new_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get('/')
def read_root():
    return {'message': 'OK'}

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('mlapi:app', port=8000, log_level='info', reload=True)
