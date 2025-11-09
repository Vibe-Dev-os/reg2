# 🎓 Student Grade Predictor

A full-stack machine learning application that predicts student final grades based on demographics, family background, study habits, and social factors using a Random Forest Regressor model.

## ✨ Features

- **🎯 Grade Prediction**: Predict student final grades (G3) based on 32 input features
- **📝 Interactive Form**: User-friendly form with validation and toast notifications
- **⚡ Real-time Results**: Instant predictions with confidence intervals
- **📊 Demographics Dashboard**: Visual analysis of student data with interactive charts
- **📈 Model Performance Metrics**: View accuracy, R², RMSE, and MAE scores
- **🌓 Dark/Light Theme**: Toggle between themes with persistent preference
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🔍 Feature Importance**: See which factors most influence predictions

## 🛠️ Tech Stack

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **Recharts** - Data visualization
- **Sonner** - Toast notifications

### Backend
- **FastAPI** - Python web framework
- **scikit-learn** - Machine learning
- **pandas** - Data processing
- **NumPy** - Numerical computing
- **Uvicorn** - ASGI server

### ML Model
- **Random Forest Regressor** with enhanced features:
  - Polynomial features (degree 2)
  - StandardScaler normalization
  - Regularization techniques
  - Correlation analysis

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download](https://www.python.org/)
- **npm** or **yarn** - Comes with Node.js
- **pip** - Comes with Python

Check your versions:
```bash
node --version
python --version
npm --version
```

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd machine-learning
```

### Step 2: Install Frontend Dependencies

```bash
npm install
```

This will install all required packages including:
- Next.js, React, TypeScript
- shadcn/ui components
- Recharts, Sonner
- And more...

### Step 3: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- FastAPI
- scikit-learn
- pandas
- numpy
- uvicorn
- joblib

**Note**: On Windows, you might need to use `pip3` instead of `pip`.

## 🏃 Running the Application

You need to run **BOTH** the backend and frontend servers.

### Step 1: Start the Backend Server

Open a terminal and run:

```bash
cd backend
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

✅ **Backend is now running at:** `http://localhost:8000`

### Step 2: Train the Model (First Time Only)

The model needs to be trained before making predictions.

**Option A: Use curl**
```bash
curl -X POST http://localhost:8000/train
```

**Option B: Use browser**
Navigate to `http://localhost:8000/docs` and use the Swagger UI to call `/train`

**Option C: Auto-train**
The model will automatically train on the first prediction attempt.

### Step 3: Start the Frontend Server

Open a **NEW terminal** (keep backend running) and run:

```bash
npm run dev
```

You should see:
```
  ▲ Next.js 15.0.1
  - Local:        http://localhost:3000
  - Ready in 2.5s
```

✅ **Frontend is now running at:** `http://localhost:3000`

### Step 4: Open in Browser

Navigate to [http://localhost:3000](http://localhost:3000)

## 📖 Usage Guide

### 1. View Dataset Statistics

On the home page, you'll see 4 cards showing:
- **Total Records**: 649 students
- **Average Grade**: 11.91/20
- **Grade Range**: 0-19 points
- **Features**: 33 data attributes

### 2. Make a Prediction

#### Navigate to "Predict Grade" Tab:

1. **Fill in all required fields** in the form:
   
   **Student Demographics:**
   - School (GP or MS)
   - Sex (Male/Female)
   - Age (15-22)
   - Address (Urban/Rural)

   **Family Background:**
   - Family size (≤3 or >3)
   - Parent status (Together/Apart)
   - Mother's education (0-4)
   - Father's education (0-4)
   - Mother's job
   - Father's job
   - Reason for school choice
   - Guardian

   **Study Habits:**
   - Travel time (1-4)
   - Study time (1-4)
   - Failures (0-4)
   - School support (Yes/No)
   - Family support (Yes/No)
   - Extra paid classes (Yes/No)

   **Social & Lifestyle:**
   - Extra-curricular activities (Yes/No)
   - Attended nursery school (Yes/No)
   - Wants higher education (Yes/No)
   - Internet at home (Yes/No)
   - Romantic relationship (Yes/No)
   - Family relationship quality (1-5)
   - Free time (1-5)
   - Going out with friends (1-5)
   - Workday alcohol consumption (1-5)
   - Weekend alcohol consumption (1-5)
   - Health status (1-5)
   - Absences (0+)

   **Previous Grades:**
   - G1 (First period grade: 0-20)
   - G2 (Second period grade: 0-20)

2. **Click "Predict Final Grade"** button

3. **View results** in the modal:
   - Predicted grade (0-20 scale)
   - Confidence interval (95%)
   - Top 5 influential features
   - Pass/Fail indicator (≥10 = Pass)

### 3. View Demographics

Click the **"Demographics"** tab to see:
- **Gender Distribution**: Male vs Female students (pie chart)
- **Age Distribution**: Student age ranges (bar chart)
- **Address Type**: Urban vs Rural (pie chart)
- **Final Grade Distribution**: Grade ranges 0-5, 6-10, 11-15, 16-20 (bar chart)
- **Family Size**: ≤3 vs >3 members (pie chart)
- **Parent Status**: Together vs Apart (pie chart)

### 4. Check Model Performance

On the right sidebar (Predict Grade tab), view:
- **Overall Accuracy**: Percentage score (~85%)
- **R² Score**: Variance explained (~0.82)
- **RMSE**: Average prediction error (~±2.0 points)
- **MAE**: Mean absolute error (~±1.5 points)

### 5. Download Dataset

Click the **"Download Dataset"** button in the header to download `student-por.csv`

### 6. Toggle Theme

Click the **moon/sun icon** in the header to switch between dark and light themes.

## 🤖 Model Information

### Dataset
- **Source**: `student-por.csv`
- **Students**: 649 Portuguese secondary school students
- **Features**: 32 input variables
- **Target**: Final grade (G3) on 0-20 scale
- **Origin**: UCI Machine Learning Repository

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Estimators**: 200 decision trees
- **Max Depth**: 15 levels
- **Features per Split**: sqrt (regularization)
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2

### Pro Tips Applied
1. ✅ **Correlation Analysis** - Identifies important features
2. ✅ **Polynomial Features** - Captures non-linear relationships (degree 2)
3. ✅ **Feature Scaling** - StandardScaler normalization
4. ✅ **Regularization** - Prevents overfitting with max_features='sqrt'

### Performance Metrics
- **Accuracy**: ~85%
- **R² Score**: ~0.82 (82% variance explained)
- **RMSE**: ~2.0 points (average error)
- **MAE**: ~1.5 points (mean absolute error)

### What This Means
- Model predicts grades within ±2 points on average
- 82% of grade variations are explained by the features
- Excellent performance for educational prediction

## 📁 Project Structure

```
machine-learning/
├── app/                          # Next.js app directory
│   ├── page.tsx                 # Main page component
│   ├── layout.tsx               # Root layout with theme
│   └── globals.css              # Global styles & Tailwind
│
├── backend/                      # FastAPI backend
│   ├── main.py                  # API endpoints & ML logic
│   ├── requirements.txt         # Python dependencies
│   └── models/                  # Trained model files (auto-generated)
│       ├── grade_predictor.pkl  # Trained Random Forest model
│       ├── scaler.pkl           # StandardScaler object
│       ├── poly_features.pkl    # Polynomial features transformer
│       ├── label_encoders.pkl   # Categorical encoders
│       ├── feature_columns.pkl  # Feature names
│       └── metrics.json         # Model performance metrics
│
├── components/                   # React components
│   ├── prediction-form.tsx      # Prediction form with validation
│   ├── model-metrics.tsx        # Performance metrics display
│   ├── demographics-charts.tsx  # Demographics visualizations
│   ├── theme-provider.tsx       # Theme context provider
│   ├── theme-toggle.tsx         # Theme switcher button
│   └── ui/                      # shadcn/ui components
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       ├── select.tsx
│       ├── tabs.tsx
│       ├── progress.tsx
│       ├── sonner.tsx           # Toast notifications
│       └── ... (more components)
│
├── lib/
│   ├── api.ts                   # API client functions
│   └── utils.ts                 # Utility functions
│
├── student-por.csv              # Dataset (649 students)
├── package.json                 # Node.js dependencies
├── tsconfig.json                # TypeScript configuration
├── tailwind.config.ts           # Tailwind CSS configuration
├── next.config.ts               # Next.js configuration
└── README.md                    # This file
```

## 🌐 API Endpoints

### Base URL
`http://localhost:8000`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status check |
| `POST` | `/train` | Train the ML model |
| `POST` | `/predict` | Make grade prediction |
| `GET` | `/model/status` | Check if model is trained |
| `GET` | `/dataset/info` | Get dataset statistics |
| `GET` | `/demographics` | Get demographic data for charts |

### Example: Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "teacher",
    "Fjob": "services",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "yes",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 2,
    "G1": 14,
    "G2": 15
  }'
```

#### Update CORS in Backend

```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app"  # Add your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🔧 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
cd backend
pip install -r requirements.txt
```

---

**Problem**: Port 8000 already in use

**Solution**:
```bash
# Find and kill the process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -ti:8000 | xargs kill -9
```

---

**Problem**: CORS errors in browser

**Solution**:
- Check `allow_origins` in `backend/main.py`
- Ensure frontend URL is in the allowed origins list
- Restart backend server after changes

### Frontend Issues

**Problem**: Cannot connect to backend

**Solution**:
1. Ensure backend is running on port 8000
2. Check `API_BASE_URL` in `lib/api.ts`
3. Verify no firewall blocking

---

**Problem**: `Module not found` errors

**Solution**:
```bash
rm -rf node_modules package-lock.json
npm install
```

---

**Problem**: Theme not persisting

**Solution**:
- Clear browser localStorage
- Check if cookies are enabled

### Model Issues

**Problem**: "Model Not Trained" error

**Solution**:
```bash
curl -X POST http://localhost:8000/train
```

---

**Problem**: Prediction validation errors

**Solution**:
- Ensure all 32 fields are filled
- Check that numeric values are in valid ranges
- Look for toast notifications showing which field is invalid

---

**Problem**: Low accuracy predictions

**Solution**:
- Retrain the model with the enhanced Pro Tips
- Ensure G1 and G2 grades are accurate
- Check that all features are correctly filled

## 📄 License

This project is for educational purposes.

## 👥 Authors

**Group 2: Regressors**

## 🙏 Acknowledgments


---

**Made with ❤️ by Group 2: Regressors**
