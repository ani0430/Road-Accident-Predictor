from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests
import folium
import pandas as pd
import os

app = Flask(__name__)
WEATHER_API_KEY = "60494532620c34f162c3b33c191627bc"

def load_model():
    with open('model/accident_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('model/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, label_encoders, feature_names

model, label_encoders, feature_names = load_model()

STATE_COORDS = {
    'Andhra Pradesh': [15.9129, 79.7400], 'Arunachal Pradesh': [28.2180, 94.7278],
    'Assam': [26.2006, 92.9376], 'Bihar': [25.0961, 85.3131],
    'Chhattisgarh': [21.2787, 81.8661], 'Goa': [15.2993, 74.1240],
    'Gujarat': [22.2587, 71.1924], 'Haryana': [29.0588, 76.0856],
    'Himachal Pradesh': [31.1048, 77.1734], 'Jammu & Kashmir': [33.7782, 76.5762],
    'Jharkhand': [23.6102, 85.2799], 'Karnataka': [15.3173, 75.7139],
    'Kerala': [10.8505, 76.2711], 'Madhya Pradesh': [22.9734, 78.6569],
    'Maharashtra': [19.7515, 75.7139], 'Manipur': [24.6637, 93.9063],
    'Meghalaya': [25.4670, 91.3662], 'Mizoram': [23.1645, 92.9376],
    'Nagaland': [26.1584, 94.5624], 'Odisha': [20.9517, 85.0985],
    'Punjab': [31.1471, 75.3412], 'Rajasthan': [27.0238, 74.2179],
    'Sikkim': [27.5330, 88.5122], 'Tamil Nadu': [11.1271, 78.6569],
    'Telangana': [18.1124, 79.0193], 'Tripura': [23.9408, 91.9882],
    'Uttarakhand': [30.0668, 79.0193], 'Uttar Pradesh': [26.8467, 80.9462],
    'West Bengal': [22.9868, 87.8550], 'Delhi': [28.7041, 77.1025],
}

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get('cod') == 200:
            weather_main = data['weather'][0]['main']
            visibility = data.get('visibility', 10000)
            if weather_main in ['Rain', 'Drizzle', 'Thunderstorm']: weather_cat = 'Rainy'
            elif weather_main in ['Fog', 'Mist', 'Haze', 'Smoke']: weather_cat = 'Foggy'
            elif weather_main == 'Clouds': weather_cat = 'Cloudy'
            else: weather_cat = 'Clear'
            if visibility >= 8000: vis_cat = 'High'
            elif visibility >= 4000: vis_cat = 'Medium'
            elif visibility >= 1000: vis_cat = 'Low'
            else: vis_cat = 'Very Low'
            return {'success': True, 'weather': weather_cat, 'visibility': vis_cat,
                'temp': data['main']['temp'], 'humidity': data['main']['humidity'],
                'wind': data['wind']['speed'],
                'description': data['weather'][0]['description'].title(),
                'icon': data['weather'][0]['icon']}
    except: pass
    return {'success': False}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weather', methods=['POST'])
def weather():
    city = request.json.get('city', 'Delhi')
    return jsonify(get_weather(city))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        weather_enc = label_encoders['weather'].transform([data['weather']])[0]
        road_type_enc = label_encoders['road_type'].transform([data['road_type']])[0]
        time_enc = label_encoders['time_of_day'].transform([data['time_of_day']])[0]
        alcohol_enc = label_encoders['alcohol'].transform([data['alcohol']])[0]
        visibility_enc = label_encoders['visibility'].transform([data['visibility']])[0]
        road_cond_enc = label_encoders['road_condition'].transform([data['road_condition']])[0]
        features = np.array([[int(data['age']), int(data['speed']), weather_enc,
                               road_type_enc, time_enc, alcohol_enc, visibility_enc, road_cond_enc]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        severity = label_encoders['accident_severity'].inverse_transform([prediction])[0]
        classes = label_encoders['accident_severity'].classes_
        prob_dict = {cls: round(float(prob)*100, 1) for cls, prob in zip(classes, probabilities)}
        tips = []
        if int(data['speed']) > 80: tips.append("⚠️ Reduce your speed — too fast increases risk.")
        if data['alcohol'] == 'Yes': tips.append("🚫 Never drive under alcohol influence.")
        if data['weather'] in ['Rainy', 'Foggy']: tips.append("🌧️ Drive slowly in bad weather.")
        if data['time_of_day'] == 'Night': tips.append("🌙 Night driving needs extra alertness.")
        if data['visibility'] == 'Very Low': tips.append("👁️ Very low visibility — stop at safe place.")
        if data['road_condition'] == 'Icy': tips.append("❄️ Icy roads — drive very slowly.")
        if not tips: tips.append("✅ Conditions seem safe — always stay alert!")
        return jsonify({'success': True, 'severity': severity, 'probabilities': prob_dict, 'tips': tips})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/map')
def accident_map():
    try:
        df = pd.read_csv('data/Road_Accidents_2017.csv')
        killed_col = 'State/UT-wise Total Number of Persons Killed in Road Accidents during - 2017'
        df = df[df['States/UTs'] != 'Total'][['States/UTs', killed_col]].copy()
        df.columns = ['state', 'killed']
        df['killed'] = pd.to_numeric(df['killed'], errors='coerce').fillna(0)
        max_killed = df['killed'].max()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB dark_matter')

    for _, row in df.iterrows():
        state = row['state']
        killed = int(row['killed'])
        if state not in STATE_COORDS or killed == 0:
            continue
        coords = STATE_COORDS[state]
        ratio = killed / max_killed
        if ratio > 0.5: color, risk = '#e63946', 'High Risk'
        elif ratio > 0.2: color, risk = '#f4a261', 'Medium Risk'
        else: color, risk = '#2a9d8f', 'Low Risk'
        radius = 8 + (ratio * 25)
        folium.CircleMarker(
            location=coords, radius=radius, color=color,
            fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(f"<b>{state}</b><br>{risk}<br>Deaths (2017): {killed:,}", max_width=200),
            tooltip=f"{risk}: {state} ({killed:,} deaths)"
        ).add_to(m)

    os.makedirs('static', exist_ok=True)
    m.save('static/map.html')
    return jsonify({'success': True})

@app.route('/stats')
def stats():
    try:
        df = pd.read_csv('data/Road_Accidents_2017.csv')
        killed_col = 'State/UT-wise Total Number of Persons Killed in Road Accidents during - 2017'
        df = df[df['States/UTs'] != 'Total'].copy()
        df['killed'] = pd.to_numeric(df[killed_col], errors='coerce').fillna(0)
        max_k = df['killed'].max()
        high = len(df[df['killed'] / max_k > 0.5])
        medium = len(df[(df['killed'] / max_k > 0.2) & (df['killed'] / max_k <= 0.5)])
        low = len(df[df['killed'] / max_k <= 0.2])
        total = int(df['killed'].sum())
        col14 = 'State/UT-wise Total Number of Persons Killed in Road Accidents during - 2014'
        col15 = 'State/UT-wise Total Number of Persons Killed in Road Accidents during - 2015'
        col16 = 'State/UT-wise Total Number of Persons Killed in Road Accidents during - 2016'
        col17 = killed_col
        yearly = [int(pd.to_numeric(df[c], errors='coerce').sum()) for c in [col14, col15, col16, col17]]
    except:
        high, medium, low, total = 8, 12, 16, 147913
        yearly = [131000, 142000, 147000, 147913]

    return jsonify({
        'total_predictions': total, 'high_risk': high,
        'medium_risk': medium, 'low_risk': low,
        'monthly_data': yearly,
        'months': ['2014', '2015', '2016', '2017']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
