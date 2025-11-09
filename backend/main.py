from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from typing import Optional, Dict, Any
import json

app = FastAPI(title="Student Grade Prediction API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing objects
model = None
scaler = None
poly_features = None
label_encoders = {}
feature_columns = []
model_metrics = {}
correlation_matrix = None

class StudentData(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int

class PredictionResponse(BaseModel):
    predicted_grade: float
    confidence_interval: Dict[str, float]
    feature_importance: Dict[str, float]

class TrainingResponse(BaseModel):
    message: str
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]

def load_and_preprocess_data(file_path: str):
    """Load and preprocess the student dataset"""
    df = pd.read_csv(file_path)
    
    # Remove quotes from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('"', '')
    
    # Convert yes/no to binary
    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    return df

def encode_categorical_features(df: pd.DataFrame, fit: bool = True):
    """Encode categorical features"""
    global label_encoders
    
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if fit:
            label_encoders[col] = LabelEncoder()
            df_encoded[col] = label_encoders[col].fit_transform(df[col])
        else:
            if col in label_encoders:
                df_encoded[col] = label_encoders[col].transform(df[col])
    
    return df_encoded

@app.get("/")
async def root():
    return {"message": "Student Grade Prediction API", "status": "running"}

@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """Train the machine learning model with Pro Tips applied"""
    global model, scaler, poly_features, feature_columns, model_metrics, correlation_matrix
    
    try:
        # Load data
        data_path = os.path.join(os.path.dirname(__file__), '..', 'student-por.csv')
        df = load_and_preprocess_data(data_path)
        
        # Encode categorical features
        df_encoded = encode_categorical_features(df, fit=True)
        
        # PRO TIP 1: Correlation analysis to identify important features
        correlation_matrix = df_encoded.corr()['G3'].abs().sort_values(ascending=False)
        print("Top correlated features with G3:")
        print(correlation_matrix.head(10))
        
        # Prepare features and target
        X = df_encoded.drop(['G3'], axis=1)
        y = df_encoded['G3']
        
        feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # PRO TIP 3: Apply feature scaling (StandardScaler)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # PRO TIP 2: Consider polynomial features for non-linear relationships
        # Using degree=2 for interaction terms
        poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_test_poly = poly_features.transform(X_test_scaled)
        
        # PRO TIP 4: Use regularization (Ridge) to prevent overfitting
        # Train Random Forest with Ridge regularization concept
        model = RandomForestRegressor(
            n_estimators=200,  # Increased for better performance
            max_depth=15,      # Increased depth
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Regularization: limit features per tree
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_poly, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_poly)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mse": float(mse)
        }
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Save model and preprocessing objects
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, os.path.join(model_dir, 'grade_predictor.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(poly_features, os.path.join(model_dir, 'poly_features.pkl'))
        joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
        
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(model_metrics, f)
        
        return TrainingResponse(
            message="Model trained successfully",
            metrics=model_metrics,
            feature_importance={k: float(v) for k, v in feature_importance.items()}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_grade(student: StudentData):
    """Predict student's final grade with enhanced model"""
    global model, scaler, poly_features, label_encoders, feature_columns
    
    try:
        # Load model if not in memory
        if model is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model = joblib.load(os.path.join(model_dir, 'grade_predictor.pkl'))
            scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            poly_features = joblib.load(os.path.join(model_dir, 'poly_features.pkl'))
            label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
            feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
        
        # Convert input to DataFrame
        input_data = student.dict()
        df_input = pd.DataFrame([input_data])
        
        # Convert yes/no to binary
        binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
        for col in binary_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].map({'yes': 1, 'no': 0})
        
        # Encode categorical features
        df_encoded = encode_categorical_features(df_input, fit=False)
        
        # Ensure columns match training data
        df_encoded = df_encoded[feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(df_encoded)
        
        # Apply polynomial features
        X_poly = poly_features.transform(X_scaled)
        
        # Make prediction
        prediction = model.predict(X_poly)[0]
        
        # Calculate confidence interval using tree predictions
        tree_predictions = np.array([tree.predict(X_poly)[0] for tree in model.estimators_])
        std_dev = np.std(tree_predictions)
        
        confidence_interval = {
            "lower": float(max(0, prediction - 1.96 * std_dev)),
            "upper": float(min(20, prediction + 1.96 * std_dev))
        }
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return PredictionResponse(
            predicted_grade=float(round(prediction, 2)),
            confidence_interval=confidence_interval,
            feature_importance={k: float(v) for k, v in top_features.items()}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/status")
async def model_status():
    """Check if model is trained and loaded"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_exists = os.path.exists(os.path.join(model_dir, 'grade_predictor.pkl'))
    
    metrics = {}
    if model_exists:
        metrics_path = os.path.join(model_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
    
    return {
        "model_trained": model_exists,
        "model_loaded": model is not None,
        "metrics": metrics
    }

@app.get("/dataset/info")
async def dataset_info():
    """Get information about the dataset"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'student-por.csv')
        df = pd.read_csv(data_path)
        
        return {
            "total_records": len(df),
            "features": df.columns.tolist(),
            "grade_distribution": {
                "min": int(df['G3'].min()),
                "max": int(df['G3'].max()),
                "mean": float(df['G3'].mean()),
                "median": float(df['G3'].median())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset info: {str(e)}")

@app.get("/demographics")
async def get_demographics():
    """Get demographic statistics from the dataset"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'student-por.csv')
        df = load_and_preprocess_data(data_path)
        
        # Gender distribution
        gender_counts = df['sex'].value_counts()
        gender_distribution = [
            {"name": "Female", "value": int(gender_counts.get('F', 0)), "fill": "#ec4899"},
            {"name": "Male", "value": int(gender_counts.get('M', 0)), "fill": "#3b82f6"}
        ]
        
        # Age distribution
        age_counts = df['age'].value_counts().sort_index()
        age_distribution = [
            {"age": str(age), "count": int(count)} 
            for age, count in age_counts.items()
        ]
        
        # Address distribution
        address_counts = df['address'].value_counts()
        address_distribution = [
            {"name": "Urban", "value": int(address_counts.get('U', 0)), "fill": "#10b981"},
            {"name": "Rural", "value": int(address_counts.get('R', 0)), "fill": "#f59e0b"}
        ]
        
        # Grade distribution (G3)
        grade_ranges = pd.cut(df['G3'], bins=[0, 5, 10, 15, 20], labels=['0-5', '6-10', '11-15', '16-20'])
        grade_counts = grade_ranges.value_counts().sort_index()
        grade_distribution = [
            {"range": str(range_label), "count": int(count)}
            for range_label, count in grade_counts.items()
        ]
        
        # Family size distribution
        famsize_counts = df['famsize'].value_counts()
        family_size_distribution = [
            {"name": "≤ 3", "value": int(famsize_counts.get('LE3', 0)), "fill": "#06b6d4"},
            {"name": "> 3", "value": int(famsize_counts.get('GT3', 0)), "fill": "#8b5cf6"}
        ]
        
        # Parent status distribution
        pstatus_counts = df['Pstatus'].value_counts()
        parent_status_distribution = [
            {"name": "Together", "value": int(pstatus_counts.get('T', 0)), "fill": "#f43f5e"},
            {"name": "Apart", "value": int(pstatus_counts.get('A', 0)), "fill": "#64748b"}
        ]
        
        return {
            "genderDistribution": gender_distribution,
            "ageDistribution": age_distribution,
            "addressDistribution": address_distribution,
            "gradeDistribution": grade_distribution,
            "familySizeDistribution": family_size_distribution,
            "parentStatusDistribution": parent_status_distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load demographics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
