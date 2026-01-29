# Wine Classification API

DADS 7305 - Machine Learning Operations (MLOps) | Lab 1

A FastAPI app that classifies wines into 3 classes based on 13 chemical features. Uses a neural network (MLPClassifier) trained on sklearn's Wine dataset.

## Folder Structure

```
├── model/
│   ├── wine_model.pkl
│   └── scaler.pkl
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv mlops-fastapi-env
mlops-fastapi-env\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run

train the model:
```bash
cd src
python train.py
```

Start the API:
```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test.

## Example Request

POST `/predict`
```json
{
  "alcohol": 14.23,
  "malic_acid": 1.71,
  "ash": 2.43,
  "alcalinity_of_ash": 15.6,
  "magnesium": 127,
  "total_phenols": 2.8,
  "flavanoids": 3.06,
  "nonflavanoid_phenols": 0.28,
  "proanthocyanins": 2.29,
  "color_intensity": 5.64,
  "hue": 1.04,
  "od280_od315_of_diluted_wines": 3.92,
  "proline": 1065
}
```

Response:
```json
{
  "response": 0
}
```

The response is the predicted wine class (0, 1, or 2).
