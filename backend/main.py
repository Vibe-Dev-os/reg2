from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel, RFE
import joblib
import os
from typing import Optional, Dict, Any
import json

app = FastAPI(title="Student Grade Prediction API")

# Debug: Print all routes on startup
@app.on_event("startup")
async def startup_event():
    print("=== FastAPI Startup ===")
    print("Available routes:")
    for route in app.routes:
        print(f"  {route.methods} {route.path}")
    print("======================")

# CORS middleware
# Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing objects
model = None
scaler = None
poly_features = None
feature_selector = None
label_encoders = {}
feature_columns = []
selected_feature_names = []
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
                try:
                    # Check if value is valid
                    value = df[col].iloc[0]
                    if value not in label_encoders[col].classes_:
                        valid_values = ', '.join(label_encoders[col].classes_)
                        raise ValueError(f"Invalid value '{value}' for field '{col}'. Valid values are: {valid_values}")
                    df_encoded[col] = label_encoders[col].transform(df[col])
                except Exception as e:
                    if "previously unseen" in str(e) or "not in" in str(e):
                        valid_values = ', '.join(label_encoders[col].classes_)
                        raise ValueError(f"Invalid value for field '{col}'. Valid values are: {valid_values}")
                    raise
    
    return df_encoded

def detect_and_remove_outliers(X, y, contamination=0.05):
    """
    Detect and remove outliers using Isolation Forest
    Returns cleaned X and y
    """
    # Combine X and y for outlier detection
    data_combined = np.column_stack([X, y])
    
    # Use Isolation Forest for outlier detection
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(data_combined)
    
    # Keep only inliers (labeled as 1)
    mask = outlier_labels == 1
    X_clean = X[mask]
    y_clean = y[mask]
    
    removed_count = len(X) - len(X_clean)
    print(f"Outliers removed: {removed_count} ({removed_count/len(X)*100:.2f}%)")
    
    return X_clean, y_clean

def select_important_features(X_train, y_train, X_test, feature_names, threshold='median'):
    """
    Select important features using Random Forest feature importance
    Returns selected features and their names
    """
    # Train a Random Forest to get feature importance
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    
    # Select features based on importance threshold
    selector = SelectFromModel(rf_selector, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"Features selected: {len(selected_features)} out of {len(feature_names)}")
    print(f"Selected features: {selected_features}")
    
    return X_train_selected, X_test_selected, selected_features

@app.get("/")
async def root():
    return {"message": "Student Grade Prediction API", "status": "running"}

@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """Train the machine learning model with Pro Tips + Improvements applied"""
    global model, scaler, poly_features, feature_selector, feature_columns, selected_feature_names, model_metrics, correlation_matrix
    
    try:
        # Load data
        data_path = os.path.join(os.path.dirname(__file__), 'student-por.csv')
        df = load_and_preprocess_data(data_path)
        
        # Encode categorical features
        df_encoded = encode_categorical_features(df, fit=True)
        
        # PRO TIP 1: Correlation analysis to identify important features
        correlation_matrix = df_encoded.corr()['G3'].abs().sort_values(ascending=False)
        print("\n" + "="*60)
        print("PRO TIP 1: CORRELATION ANALYSIS")
        print("="*60)
        print("Top correlated features with G3:")
        print(correlation_matrix.head(10))
        
        # Prepare features and target
        X = df_encoded.drop(['G3'], axis=1)
        y = df_encoded['G3']
        
        feature_columns = X.columns.tolist()
        
        # IMPROVEMENT 1: Handle outliers using Isolation Forest
        print("\n" + "="*60)
        print("IMPROVEMENT 1: OUTLIER DETECTION & REMOVAL")
        print("="*60)
        X_clean, y_clean = detect_and_remove_outliers(X.values, y.values, contamination=0.05)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # PRO TIP 3: Apply feature scaling (StandardScaler)
        print("\n" + "="*60)
        print("PRO TIP 3: FEATURE SCALING")
        print("="*60)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Applied StandardScaler normalization")
        
        # IMPROVEMENT 2: Feature selection to reduce overfitting
        print("\n" + "="*60)
        print("IMPROVEMENT 2: FEATURE SELECTION")
        print("="*60)
        X_train_selected, X_test_selected, selected_feature_names = select_important_features(
            X_train_scaled, y_train, X_test_scaled, feature_columns, threshold='median'
        )
        
        # PRO TIP 2: Consider polynomial features for non-linear relationships
        print("\n" + "="*60)
        print("PRO TIP 2: POLYNOMIAL FEATURES")
        print("="*60)
        # Using degree=2 for interaction terms
        poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_train_poly = poly_features.fit_transform(X_train_selected)
        X_test_poly = poly_features.transform(X_test_selected)
        print(f"Generated polynomial features: {X_train_poly.shape[1]} features")
        
        # PRO TIP 4: Use regularization (Ridge) to prevent overfitting
        print("\n" + "="*60)
        print("PRO TIP 4: REGULARIZATION & MODEL TRAINING")
        print("="*60)
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
        print("Random Forest model trained successfully")
        
        # IMPROVEMENT 3: Cross-validation for robust metrics
        print("\n" + "="*60)
        print("IMPROVEMENT 3: CROSS-VALIDATION")
        print("="*60)
        cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, 
                                    scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse_scores = np.sqrt(-cv_scores)
        print(f"5-Fold CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")
        
        # Evaluate model on test set
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        y_pred = model.predict(X_test_poly)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test Set Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        
        model_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mse": float(mse),
            "cv_rmse_mean": float(cv_rmse_scores.mean()),
            "cv_rmse_std": float(cv_rmse_scores.std())
        }
        
        # Feature importance (using selected features)
        feature_importance = dict(zip(selected_feature_names, model.feature_importances_[:len(selected_feature_names)]))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        print(f"\nTop 10 Important Features:")
        for feat, imp in feature_importance.items():
            print(f"  {feat}: {imp:.4f}")
        
        # Save model and preprocessing objects
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, os.path.join(model_dir, 'grade_predictor.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(poly_features, os.path.join(model_dir, 'poly_features.pkl'))
        joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
        joblib.dump(selected_feature_names, os.path.join(model_dir, 'selected_features.pkl'))
        
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(model_metrics, f)
        
        print("\n" + "="*60)
        print("MODEL SAVED SUCCESSFULLY")
        print("="*60)
        print(f"All model artifacts saved to: {model_dir}")
        
        return TrainingResponse(
            message="Model trained successfully with improvements: outlier removal, feature selection, and cross-validation",
            metrics=model_metrics,
            feature_importance={k: float(v) for k, v in feature_importance.items()}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_grade(student: StudentData):
    """Predict student's final grade with improved model"""
    global model, scaler, poly_features, label_encoders, feature_columns, selected_feature_names
    
    try:
        # Load model if not in memory
        if model is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model = joblib.load(os.path.join(model_dir, 'grade_predictor.pkl'))
            scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            poly_features = joblib.load(os.path.join(model_dir, 'poly_features.pkl'))
            label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
            feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
            
            # Try to load selected features (for improved model)
            selected_features_path = os.path.join(model_dir, 'selected_features.pkl')
            if os.path.exists(selected_features_path):
                selected_feature_names = joblib.load(selected_features_path)
            else:
                # Fallback: use all features if model was trained with old code
                selected_feature_names = feature_columns
        
        # Convert input to DataFrame
        input_data = student.dict()
        
        # Debug: Print received data
        print(f"\n[PREDICT] Received data: {input_data}")
        
        df_input = pd.DataFrame([input_data])
        
        # Convert yes/no to binary
        binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
        for col in binary_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].map({'yes': 1, 'no': 0})
        
        # Debug: Check categorical values before encoding
        categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
        print(f"[PREDICT] Categorical values:")
        for col in categorical_cols:
            if col in df_input.columns:
                print(f"  {col}: {df_input[col].iloc[0]}")
        
        # Encode categorical features
        df_encoded = encode_categorical_features(df_input, fit=False)
        
        # Ensure columns match training data
        df_encoded = df_encoded[feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(df_encoded)
        
        # Select only the important features used during training
        # Check if feature selection was used
        if len(selected_feature_names) < len(feature_columns):
            X_selected = X_scaled[:, [feature_columns.index(f) for f in selected_feature_names]]
        else:
            X_selected = X_scaled
        
        # Apply polynomial features
        X_poly = poly_features.transform(X_selected)
        
        # Make prediction
        prediction = model.predict(X_poly)[0]
        
        # Calculate confidence interval using tree predictions
        tree_predictions = np.array([tree.predict(X_poly)[0] for tree in model.estimators_])
        std_dev = np.std(tree_predictions)
        
        confidence_interval = {
            "lower": float(max(0, prediction - 1.96 * std_dev)),
            "upper": float(min(20, prediction + 1.96 * std_dev))
        }
        
        # Get feature importance for this prediction (using selected features)
        feature_importance = dict(zip(selected_feature_names, model.feature_importances_[:len(selected_feature_names)]))
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
        data_path = os.path.join(os.path.dirname(__file__), 'student-por.csv')
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
        data_path = os.path.join(os.path.dirname(__file__), 'student-por.csv')
        df = pd.read_csv(data_path)  # Use direct pandas read instead of preprocessing
        
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
        import traceback
        error_details = traceback.format_exc()
        print(f"Demographics error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to load demographics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
