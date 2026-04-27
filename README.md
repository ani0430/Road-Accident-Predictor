# 🚗 Road Accident Risk Predictor
### CS Major Project — AI/ML Based Road Safety System

---

## 📌 Project Overview
An AI-powered web application that predicts the risk level of road accidents based on driving conditions like speed, weather, road type, time of day, and more.

---

## 🛠️ Tech Stack
- **Python** — Core language
- **Scikit-learn** — Random Forest ML Model
- **Flask** — Web Framework
- **Pandas/NumPy** — Data Processing
- **Chart.js** — Data Visualization
- **HTML/CSS/JS** — Frontend

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```

### Step 3 — Run the app
```bash
python app.py
```

### Step 4 — Open browser
```
http://localhost:5000
```

---

## 📁 Project Structure
```
road_accident_predictor/
├── app.py                  # Flask web application
├── train_model.py          # ML model training
├── requirements.txt        # Dependencies
├── data/
│   └── accidents.csv       # Dataset
├── model/
│   ├── accident_model.pkl  # Trained ML model
│   └── label_encoders.pkl  # Encoders
└── templates/
    └── index.html          # Frontend UI
```

---

## 🎯 Features
- Real-time accident risk prediction (High/Medium/Low)
- Probability breakdown for each risk level
- Safety recommendations based on conditions
- Monthly trend analysis chart
- High-risk zone hotspots map

---

## 🧠 ML Model
- **Algorithm**: Random Forest Classifier
- **Features**: Age, Speed, Weather, Road Type, Time, Alcohol, Visibility, Road Condition
- **Output**: Risk Level — High / Medium / Low

---

## 👨‍💻 Developed By
**Animesh Jha** — CS Major Project
