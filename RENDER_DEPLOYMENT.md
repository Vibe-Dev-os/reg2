# 🚀 Deploy Backend to Render

Complete guide to deploy your FastAPI backend to Render.

---

## 📋 Prerequisites

1. ✅ GitHub account
2. ✅ Render account (free) - [Sign up here](https://render.com)
3. ✅ Your code pushed to GitHub

---

## 🔧 Step 1: Prepare Your Repository

### 1.1 Push Backend to GitHub

If you haven't already, push your code to GitHub:

```bash
cd c:\xampp\htdocs\machine\machine-learning
git init
git add .
git commit -m "Initial commit - Student Grade Predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 1.2 Verify Files

Make sure these files exist in your `backend/` folder:
- ✅ `main.py` - Your FastAPI application
- ✅ `requirements.txt` - Python dependencies
- ✅ `render.yaml` - Render configuration (just created)
- ✅ `student-por.csv` - Dataset (in parent directory)

---

## 🌐 Step 2: Deploy to Render

### Option A: Using render.yaml (Recommended)

1. **Go to Render Dashboard**
   - Visit [https://dashboard.render.com](https://dashboard.render.com)
   - Sign in or create account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your repository

3. **Configure Service**
   - **Name:** `student-grade-predictor-api`
   - **Region:** Oregon (US West) or closest to you
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Environment Variables** (Optional)
   - Click "Advanced"
   - Add if needed:
     - `PYTHON_VERSION`: `3.12.0`

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment

### Option B: Manual Configuration

If render.yaml doesn't work, use these settings:

**Service Settings:**
```
Name: student-grade-predictor-api
Environment: Python 3
Region: Oregon (US West)
Branch: main
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Advanced Settings:**
```
Auto-Deploy: Yes
Health Check Path: /
```

---

## 📝 Step 3: Configure Build Settings

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Important:** Render automatically sets the `$PORT` environment variable.

---

## 🔍 Step 4: Verify Deployment

### 4.1 Check Deployment Status

In Render dashboard:
- ✅ Build logs should show successful installation
- ✅ Service should show "Live" status
- ✅ You'll get a URL like: `https://student-grade-predictor-api.onrender.com`

### 4.2 Test API Endpoints

Open your browser or use curl:

**Test Root Endpoint:**
```bash
curl https://YOUR_APP_NAME.onrender.com/
```

Expected response:
```json
{
  "message": "Student Grade Prediction API",
  "status": "running"
}
```

**Test Docs:**
Visit: `https://YOUR_APP_NAME.onrender.com/docs`

**Test Model Status:**
```bash
curl https://YOUR_APP_NAME.onrender.com/model/status
```

---

## 🎯 Step 5: Train the Model on Render

### Important: Dataset Location

The model needs `student-por.csv`. Make sure it's in the correct location:

**Option 1: Include in Repository**
```
machine-learning/
├── backend/
│   ├── main.py
│   └── requirements.txt
└── student-por.csv  ← Dataset here
```

Update `main.py` line 173:
```python
data_path = os.path.join(os.path.dirname(__file__), '..', 'student-por.csv')
```

**Option 2: Upload via API** (if dataset is large)
Create an upload endpoint or use environment variables.

### Train the Model

Once deployed, train the model:

```bash
curl -X POST https://YOUR_APP_NAME.onrender.com/train
```

This will:
- Load the dataset
- Apply all improvements (outlier removal, feature selection, etc.)
- Train the Random Forest model
- Save model artifacts

**Note:** First training might take 1-2 minutes.

---

## 🔗 Step 6: Update Frontend to Use Render API

### Update API Base URL

In your frontend `lib/api.ts`:

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 
                     "https://YOUR_APP_NAME.onrender.com";
```

Or create `.env.local`:
```
NEXT_PUBLIC_API_URL=https://YOUR_APP_NAME.onrender.com
```

---

## ⚙️ Step 7: Environment Variables (Optional)

If you need to add environment variables:

1. Go to Render Dashboard
2. Select your service
3. Click "Environment"
4. Add variables:
   - `PYTHON_VERSION`: `3.12.0`
   - `CORS_ORIGINS`: Your frontend URL

---

## 🐛 Troubleshooting

### Issue 1: Build Fails

**Error:** `ModuleNotFoundError`

**Solution:**
- Check `requirements.txt` has all dependencies
- Verify Python version compatibility

### Issue 2: Application Crashes

**Error:** `Address already in use`

**Solution:**
- Ensure start command uses `$PORT` variable:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```

### Issue 3: CORS Errors

**Error:** `CORS policy: No 'Access-Control-Allow-Origin'`

**Solution:**
- Update CORS origins in `main.py` (already done)
- Add your frontend URL to allowed origins

### Issue 4: Model Not Found

**Error:** `Model not trained`

**Solution:**
- Train the model via `/train` endpoint
- Ensure `student-por.csv` is accessible
- Check file paths in code

### Issue 5: Slow Cold Starts

**Issue:** First request takes 30+ seconds

**Explanation:**
- Render free tier spins down after 15 minutes of inactivity
- First request "wakes up" the service

**Solutions:**
- Upgrade to paid tier ($7/month) for always-on
- Use a cron job to ping every 10 minutes
- Accept the cold start on free tier

---

## 📊 Monitoring

### View Logs

In Render Dashboard:
1. Select your service
2. Click "Logs" tab
3. See real-time application logs

### Check Metrics

Monitor:
- ✅ CPU usage
- ✅ Memory usage
- ✅ Request count
- ✅ Response times

---

## 💰 Pricing

### Free Tier
- ✅ 750 hours/month
- ✅ Automatic sleep after 15 min inactivity
- ✅ 512 MB RAM
- ✅ 0.1 CPU
- ⚠️ Cold starts (30s delay)

### Starter Tier ($7/month)
- ✅ Always on (no cold starts)
- ✅ 512 MB RAM
- ✅ 0.5 CPU
- ✅ Better performance

---

## 🔒 Security Best Practices

### 1. Restrict CORS Origins

After deployment, update `main.py`:

```python
allow_origins=[
    "https://your-frontend-domain.vercel.app",
    "http://localhost:3000",  # For local development
]
```

### 2. Add Rate Limiting

Install:
```bash
pip install slowapi
```

Add to `main.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_grade(request: Request, student: StudentData):
    # ... existing code
```

### 3. Environment Variables

Don't hardcode sensitive data. Use Render's environment variables.

---

## 🚀 Deployment Checklist

Before deploying:

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` is complete
- [ ] `render.yaml` is configured
- [ ] CORS origins updated
- [ ] Dataset is accessible
- [ ] Start command uses `$PORT`
- [ ] Tested locally

After deploying:

- [ ] Service shows "Live" status
- [ ] API root endpoint works
- [ ] `/docs` page loads
- [ ] Model trained successfully
- [ ] Frontend can connect
- [ ] Predictions work

---

## 📚 Useful Commands

### Check Service Status
```bash
curl https://YOUR_APP_NAME.onrender.com/
```

### Train Model
```bash
curl -X POST https://YOUR_APP_NAME.onrender.com/train
```

### Get Model Status
```bash
curl https://YOUR_APP_NAME.onrender.com/model/status
```

### Make Prediction
```bash
curl -X POST https://YOUR_APP_NAME.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_student.json
```

---

## 🎉 Success!

Your backend is now deployed! You should have:

✅ Live API at `https://YOUR_APP_NAME.onrender.com`  
✅ Interactive docs at `https://YOUR_APP_NAME.onrender.com/docs`  
✅ Trained model ready for predictions  
✅ Auto-deploy on git push  

---

## 🔗 Next Steps

1. **Deploy Frontend** to Vercel/Netlify
2. **Update API URL** in frontend
3. **Test end-to-end** functionality
4. **Monitor performance** in Render dashboard
5. **Set up custom domain** (optional)

---

## 📞 Support

**Render Documentation:** https://render.com/docs  
**FastAPI Deployment:** https://fastapi.tiangolo.com/deployment/  
**Render Community:** https://community.render.com/  

---

**Made with ❤️ by Group 2: Regressors**
