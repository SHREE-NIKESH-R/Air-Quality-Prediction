INSTRUCTIONS = """# 📋 COMPLETE SETUP INSTRUCTIONS

## Step 1: Install Python

Make sure Python 3.8 or higher is installed.
Check: `python --version`

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Download Dataset

1. Go to: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Download the dataset (need free Kaggle account)
3. Extract and copy `city_day.csv` to this folder

## Step 4: Train the Model

```bash
python model_training.py
```

This will take 2-3 minutes and create:

- aqi_model.pkl
- scaler.pkl
- feature_names.pkl

## Step 5: Run the Dashboard

```bash
streamlit run app.py
```

Dashboard opens at: http://localhost:8501

## 🎯 Testing

Try these values:

- PM2.5: 50, PM10: 100, NO2: 40, CO: 1.0, SO2: 15, O3: 50
- Expected AQI: ~70-80 (Moderate)

## 🌐 Deploy (Optional)

1. Push to GitHub
2. Go to streamlit.io/cloud
3. Connect repo and deploy
4. Get free live URL!

## ❓ Need Help?

Check README.md or contact your instructor.
