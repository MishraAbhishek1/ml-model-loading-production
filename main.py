
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import joblib


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load the model when this application process starts
    app.state.model = joblib.load("model.pkl")

    print("AI model loaded successfully!")

    yield

    # Application shutdown
    app.state.model = None
    print("Application shutting down")


app = FastAPI(lifespan=lifespan)


class PredictionInput(BaseModel):
    text: str


@app.post("/predict")
def predict(data: PredictionInput):

    model = app.state.model

    prediction = model.predict([data.text])

    return {
        "input": data.text,
        "prediction": prediction[0],
    }