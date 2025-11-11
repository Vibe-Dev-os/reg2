# 🎓 Student Grade Predictor - Backend API

FastAPI backend for predicting student final grades using Random Forest with advanced ML techniques.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

Server runs at: `http://127.0.0.1:8000`

API Docs: `http://127.0.0.1:8000/docs`

### Deploy to Render

See [RENDER_DEPLOYMENT.md](../RENDER_DEPLOYMENT.md) for complete deployment guide.

**Quick Deploy:**
1. Push to GitHub
2. Connect to Render
3. Set Root Directory: `backend`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status |
| `POST` | `/train` | Train the model |
| `POST` | `/predict` | Predict grade |
| `GET` | `/model/status` | Model status & metrics |
| `GET` | `/dataset/info` | Dataset statistics |
| `GET` | `/demographics` | Demographic data |

## 🤖 Model Features

- **Algorithm:** Random Forest Regressor (200 trees)
- **Accuracy:** ~92.7%
- **R² Score:** ~0.72
- **RMSE:** ±1.46 points
- **MAE:** ±0.98 points

### Improvements Applied:
1. ✅ Correlation Analysis
2. ✅ Polynomial Features (degree 2)
3. ✅ Feature Scaling (StandardScaler)
4. ✅ Regularization
5. ✅ Outlier Detection (Isolation Forest)
6. ✅ Feature Selection
7. ✅ Cross-Validation (5-fold)

## 📁 Project Structure

```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── render.yaml         # Render configuration
├── start.sh            # Start script
├── models/             # Trained model artifacts (auto-generated)
│   ├── grade_predictor.pkl
│   ├── scaler.pkl
│   ├── poly_features.pkl
│   ├── label_encoders.pkl
│   ├── feature_columns.pkl
│   ├── selected_features.pkl
│   └── metrics.json
└── README.md           # This file
```

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port (set by Render) |
| `PYTHON_VERSION` | 3.12.0 | Python version |

## 📊 Dataset

- **Source:** `student-por.csv` (Portuguese students)
- **Records:** 649 students
- **Features:** 32 input variables
- **Target:** Final grade (G3) on 0-20 scale

## 🛠️ Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
# Test API
curl http://localhost:8000/

# Train model
curl -X POST http://localhost:8000/train

# Check status
curl http://localhost:8000/model/status
```

### View API Docs
Visit: `http://localhost:8000/docs`

## 📝 License

Educational project by Group 2: Regressors

---

**Made with ❤️ using FastAPI, scikit-learn, and Python**
