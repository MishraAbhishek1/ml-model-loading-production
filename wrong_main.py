from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class PredictionInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: PredictionInput):
    
    # Loads the model on every request
    model = joblib.load("model.pkl")
    
    prediction = model.predict([data.text])
    
    return {
        "input": data.text,
        "prediction": prediction[0]
    }