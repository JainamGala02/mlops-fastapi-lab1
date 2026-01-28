import joblib
import numpy as np
import os
from data import WineData

model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
model = joblib.load(os.path.join(model_dir, 'wine_model.pkl'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))


def predict_wine(data: WineData):
    features = np.array([[
        data.alcohol,
        data.malic_acid,
        data.ash,
        data.alcalinity_of_ash,
        data.magnesium,
        data.total_phenols,
        data.flavanoids,
        data.nonflavanoid_phenols,
        data.proanthocyanins,
        data.color_intensity,
        data.hue,
        data.od280_od315_of_diluted_wines,
        data.proline
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return int(prediction[0])