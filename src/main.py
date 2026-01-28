from fastapi import FastAPI
from data import WineData, WineResponse
from predict import predict_wine

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Wine Classification API"}


@app.post("/predict", response_model=WineResponse)
async def predict(wine_data: WineData):
    prediction = predict_wine(wine_data)
    return WineResponse(response=prediction)