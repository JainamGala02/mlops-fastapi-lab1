# Wine Classification API

DADS 7305 - Machine Learning Operations (MLOps) | Lab 1

> **Note:** The original lab uses the Iris dataset with a Decision Tree Classifier. For my implementation, I've swapped in sklearn's Wine dataset (3 classes, 13 chemical features) and replaced the Decision Tree with an MLPClassifier (neural network). The FastAPI serving structure and API patterns remain the same - refer to the [original lab README](https://www.mlwithramin.com/blog/fastapi-lab1) for foundational concepts.

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

## Setup & Run

**1. Clone the repo and navigate to the project directory:**

```bash
git clone <repo-url>
cd fastapi_lab1
```

**2. Create and activate a virtual environment:**

```bash
python -m venv mlops-fastapi-env
```

- Windows: `mlops-fastapi-env\Scripts\activate`
- macOS/Linux: `source mlops-fastapi-env/bin/activate`

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

This installs `fastapi[all]` (which includes `uvicorn`), `scikit-learn`, and `joblib`.

**4. Train the model:**

```bash
cd src
python train.py
```

This trains an MLPClassifier on the Wine dataset and saves `wine_model.pkl` and `scaler.pkl` to the `model/` directory. You should see training accuracy printed to the console.

**5. Start the API server:**

```bash
uvicorn main:app --reload
```

Run this from inside the `src/` folder. The `--reload` flag enables hot-reloading during development - the server restarts automatically when you edit code.

**6. Test the API:**

- Open http://127.0.0.1:8000/docs in your browser to access the Swagger UI.
- Click on the `POST /predict` endpoint → **Try it out** → paste the example JSON below → **Execute**.
- Alternatively, use `curl` from a separate terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Expected Response

```json
{
  "response": 0
}
```

The response is the predicted wine class (0, 1, or 2).
