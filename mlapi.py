from fastapi import FastAPI, HTTPException
import pandas as pd
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
    crop: str
    temperature: float
    humidity: float
    ph: float
    water_availability: float
    country: str
    
@app.post('/predict')
def predict(data: FeaturesInput):
    try:
        new_data = pd.DataFrame({
            'temperature': [data.temperature],
            'humidity': [data.humidity],
            'ph': [data.ph],
            'water_availability': [data.water_availability],
            'crop': [data.crop],
            'country': [data.country]
        })

        new_data_enc = encoder.transform(new_data[['crop', 'country']])
        encoded_df = pd.DataFrame(new_data_enc, columns=encoder.get_feature_names_out(['crop', 'country']))

        data = pd.concat([new_data, encoded_df], axis=1)

        data = data.drop(['crop', 'country'], axis=1)

        prediction = model.predict(data)

        return {'prediction': new_data_enc}
    except Exception as e:
        raise HTTPException(status_code=500, detail='something is wrong')
    
    
@app.get('/')
def read_root():
    return {'message': 'OK'}

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('mlapi:app', host='127.0.0.1', port=8000, log_level='info', reload=True)
