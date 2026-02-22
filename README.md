# 🏠 Sri Lanka Property Price Predictor

A Machine Learning system for predicting residential property prices in Sri Lanka using CatBoost Regressor with Explainable AI (SHAP).

## 📋 Project Overview

| Item         | Details                                                    |
| ------------ | ---------------------------------------------------------- |
| **Dataset**  | 15,327 property listings from ikman.lk (Sri Lanka)         |
| **Model**    | CatBoost Regressor                                         |
| **R² Score** | 0.8415                                                     |
| **MAE**      | 5.68 Million LKR                                           |
| **Features** | Bedrooms, Bathrooms, Land Size, House Size, City, District |
| **XAI**      | SHAP Summary Plot, Feature Importance                      |

## 📁 Project Structure

```
Sri-Lanka-Property-Predictor/
├── data/
│   ├── house_prices.csv          # Raw dataset (15,327 listings)
│   └── cleaned_data.csv          # Preprocessed dataset (15,003 records)
├── models/
│   ├── catboost_model.pkl        # Trained CatBoost model
│   └── preprocessor.pkl          # Label encoders & feature config
├── notebooks/
│   ├── preprocess_and_train.py   # Full ML pipeline script
│   ├── feature_importance.png    # Feature importance visualization
│   ├── shap_summary.png          # SHAP analysis plot
│   ├── shap_bar.png              # SHAP bar chart
│   ├── actual_vs_predicted.png   # Actual vs predicted scatter
│   └── residual_distribution.png # Prediction error distribution
├── backend/
│   └── main.py                   # FastAPI prediction API
├── frontend/
│   └── index.html                # Web UI for predictions
├── scraper/
│   └── scraper.py                # Web scraper (lankapropertyweb.com)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/booshananamudara/Sri-Lanka-Property-Predictor.git
cd Sri-Lanka-Property-Predictor

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python notebooks/preprocess_and_train.py
```

This will:

1. Load and clean the raw dataset
2. Train the CatBoost Regressor
3. Evaluate the model (RMSE, MAE, R²)
4. Generate SHAP explainability plots
5. Save the model and preprocessor as `.pkl` files

### Run the Backend

```bash
uvicorn backend.main:app --reload
```

Backend runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Run the Frontend

```bash
cd frontend
python -m http.server 5500
```

Open: [http://localhost:5500](http://localhost:5500)

## 🐳 Running with Docker

```bash
docker-compose up --build
```

- Frontend: [http://localhost:5500](http://localhost:5500)
- Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📊 Model Performance

| Metric | Training  | Testing    |
| ------ | --------- | ---------- |
| MAE    | 4.58M LKR | 5.68M LKR  |
| RMSE   | 8.89M LKR | 11.72M LKR |
| R²     | 0.9009    | 0.8415     |

### Feature Importance (CatBoost)

| Feature             | Importance |
| ------------------- | ---------- |
| City                | 27.87      |
| House Size (sqft)   | 23.67      |
| Land Size (perches) | 22.54      |
| Bathrooms           | 13.41      |
| Bedrooms            | 7.11       |
| District            | 5.40       |

## 🔍 API Usage

### Predict Price

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2,
    "land_size_perches": 10,
    "house_size_sqft": 2000,
    "city": "Nugegoda",
    "district": "Colombo"
  }'
```

### Response

```json
{
  "predicted_price": 36971909.14,
  "predicted_price_formatted": "Rs. 36,971,909",
  "predicted_price_millions": 36.97
}
```

## 🧠 Explainable AI (XAI)

The project uses **SHAP (SHapley Additive exPlanations)** to explain model predictions:

- **SHAP Summary Plot** — shows the impact of each feature on predictions
- **Feature Importance** — CatBoost's built-in feature ranking
- **Actual vs Predicted** — scatter plot showing prediction accuracy
- **Residual Distribution** — histogram of prediction errors

## 📝 License

This project was created for academic purposes (MSc in AI - Machine Learning Assignment).
