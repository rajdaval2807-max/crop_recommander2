import os
import numpy as np
import joblib
import requests
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import gdown
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

files = {
    "soil_model.h5": "GOOGLE_DRIVE_FILE_ID_1",
    "crop_model.pkl": "GOOGLE_DRIVE_FILE_ID_2",
    "soil_encoder.pkl": "GOOGLE_DRIVE_FILE_ID_3"
}

for name, file_id in files.items():
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)


app = Flask(__name__)
app.secret_key = "secretkey"

# ================= DATABASE CONFIG =================
# ================= DATABASE CONFIG =================
db_path = os.path.join(os.environ.get("RENDER_DISK_PATH", ""), "agro_aiab.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ================= DATABASE TABLES =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    prediction_type = db.Column(db.String(50))
    result = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    entry_type = db.Column(db.String(20))  # 'expense' or 'income'
    name = db.Column(db.String(200))
    description = db.Column(db.Text)
    amount = db.Column(db.Float)
    date = db.Column(db.Date)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ================= WEATHER API CONFIG =================
#==WEATHER_API_KEY = "cfc8d8215798a536ee7d5c30924cd617"
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")#   # Replace with your API key
WEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

# ================= LOAD MODELS =================
crop_model = joblib.load("models/crop_model.pkl")
soil_encoder = joblib.load("models/soil_encoder.pkl")
soil_model = load_model("models/soil_model.h5")


# ================= SOIL CLASSES (CNN OUTPUT ORDER) =================
soil_classes = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil"
]

# 🔥 Mapping CNN soil names → Crop model soil names
SOIL_LABEL_MAP = {
    "Alluvial_Soil": "Alluvial Soil",
    "Arid_Soil": "Arid Soil",
    "Black_Soil": "Black Soil",
    "Laterite_Soil": "Laterite Soil",
    "Mountain_Soil": "Mountain Soil",
    "Red_Soil": "Red Soil",
    "Yellow_Soil": "Yellow Soil"
}

# ================= GOVERNMENT SCHEMES DATA =================
GOVERNMENT_SCHEMES = [
    {
        "name": "Pradhan Mantri Krishi Sinchai Yojana",
        "description": "Provides irrigation facilities to every farm",
        "eligibility": "All farmers",
        "benefits": "Up to 50% subsidy on irrigation systems",
        "link": "https://pmksy.gov.in"
    },
    {
        "name": "Soil Health Card Scheme",
        "description": "Provides soil health cards to farmers",
        "eligibility": "All landholding farmers",
        "benefits": "Free soil testing and recommendations",
        "link": "https://soilhealth.dac.gov.in"
    },
    {
        "name": "National Mission for Sustainable Agriculture",
        "description": "Promotes sustainable agriculture practices",
        "eligibility": "Farmers practicing sustainable methods",
        "benefits": "Financial assistance and training",
        "link": "https://nmsa.dac.gov.in"
    },
    {
        "name": "Paramparagat Krishi Vikas Yojana",
        "description": "Promotes organic farming",
        "eligibility": "Farmers practicing organic farming",
        "benefits": "₹50,000 per hectare for 3 years",
        "link": "https://pgsindia-ncof.gov.in"
    },
    {
        "name": "PM Kisan Samman Nidhi",
        "description": "Direct income support to farmers",
        "eligibility": "All small and marginal farmers",
        "benefits": "₹6,000 per year in three installments",
        "link": "https://pmkisan.gov.in"
    }
]

# ================= HELPER FUNCTIONS =================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def predict_soil(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = soil_model.predict(img_array)
    class_index = np.argmax(predictions)
    return soil_classes[class_index]

def analyze_soil_health(N, P, K, ph):
    """Analyze soil health based on NPK and pH values"""
    score = 0
    suggestions = []
    
    # Nitrogen analysis
    if N > 80:
        score += 25
    elif N > 50:
        score += 15
        suggestions.append("Add nitrogen-rich fertilizer (Urea/Ammonium)")
    else:
        score += 5
        suggestions.append("Urgent: Add nitrogen fertilizer immediately")
    
    # Phosphorus analysis
    if P > 40:
        score += 25
    elif P > 20:
        score += 15
        suggestions.append("Add phosphorus fertilizer (DAP/Superphosphate)")
    else:
        score += 5
        suggestions.append("Urgent: Add phosphorus fertilizer")
    
    # Potassium analysis
    if K > 40:
        score += 25
    elif K > 20:
        score += 15
        suggestions.append("Add potassium fertilizer (MOP/SOP)")
    else:
        score += 5
        suggestions.append("Urgent: Add potassium fertilizer")
    
    # pH analysis
    if 6.0 <= ph <= 7.5:
        score += 25
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        score += 15
        if ph < 6.0:
            suggestions.append("Add lime to increase pH")
        else:
            suggestions.append("Add sulfur to decrease pH")
    else:
        score += 5
        suggestions.append("Soil pH is critical. Consult expert immediately")
    
    # Determine health level
    if score >= 80:
        health = "Excellent"
        color = "green"
    elif score >= 60:
        health = "Good"
        color = "blue"
    elif score >= 40:
        health = "Moderate"
        color = "orange"
    else:
        health = "Poor"
        color = "red"
    
    return {
        "score": score,
        "health": health,
        "color": color,
        "suggestions": suggestions
    }

def get_weather_data(city):
    """Fetch 5-day weather forecast from OpenWeather API"""
    try:
        params = {
            'q': city,
            'appid': WEATHER_API_KEY,
            'units': 'metric',
            'cnt': 40  # 5 days * 8 periods per day
        }
        response = requests.get(WEATHER_BASE_URL, params=params)
        data = response.json()
        
        if data.get("cod") != "200":
            return None
        
        # Process 5-day forecast
        forecast = []
        warnings = []
        
        # Group by day
        daily_data = {}
        for item in data['list']:
            date = item['dt_txt'].split()[0]
            if date not in daily_data:
                daily_data[date] = {
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'humidity': item['main']['humidity'],
                    'rain': item.get('rain', {}).get('3h', 0),
                    'wind_speed': item['wind']['speed'],
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon']
                }
            else:
                daily_data[date]['temp_min'] = min(daily_data[date]['temp_min'], item['main']['temp_min'])
                daily_data[date]['temp_max'] = max(daily_data[date]['temp_max'], item['main']['temp_max'])
                daily_data[date]['rain'] += item.get('rain', {}).get('3h', 0)
        
        # Convert to list and check for warnings
        for date_str, day_data in list(daily_data.items())[:5]:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            forecast.append({
                'date': date_obj,
                'day_name': date_obj.strftime('%A'),
                'temp_min': round(day_data['temp_min'], 1),
                'temp_max': round(day_data['temp_max'], 1),
                'humidity': day_data['humidity'],
                'rain': round(day_data['rain'], 2),
                'wind_speed': round(day_data['wind_speed'], 1),
                'description': day_data['description'],
                'icon': day_data['icon']
            })
            
            # Check for warnings
            if day_data['rain'] > 20:  # Heavy rainfall
                warnings.append(f"Heavy rainfall ({day_data['rain']}mm) expected on {date_obj.strftime('%b %d')}. Consider delaying irrigation and protecting crops.")
            if day_data['temp_max'] > 40:  # Heatwave
                warnings.append(f"Heatwave warning! Temperature up to {day_data['temp_max']}°C on {date_obj.strftime('%b %d')}. Increase irrigation and provide shade.")
            if day_data['wind_speed'] > 30:  # Strong winds
                warnings.append(f"Strong winds ({day_data['wind_speed']} km/h) on {date_obj.strftime('%b %d')}. Secure crops and equipment.")
        
        return {
            'city': data['city']['name'],
            'country': data['city']['country'],
            'forecast': forecast,
            'warnings': warnings
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

# ================= ROUTES =================
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(username=username).first():
            return "User already exists!"
        db.session.add(User(username=username, password=password))
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user"] = username
            return redirect(url_for("dashboard"))
        return "Invalid credentials"
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    # Get user's recent history
    recent_history = History.query.filter_by(username=session["user"])\
        .order_by(History.timestamp.desc())\
        .limit(3).all()
    
    # Get total expenses and income
    expenses = Expense.query.filter_by(
        username=session["user"],
        entry_type='expense'
    ).all()
    income = Expense.query.filter_by(
        username=session["user"],
        entry_type='income'
    ).all()
    
    total_expense = sum([e.amount for e in expenses])
    total_income = sum([i.amount for i in income])
    
    return render_template("dashboard.html",
                         recent_history=recent_history,
                         total_expense=total_expense,
                         total_income=total_income)

# ================= CROP PREDICTION =================
@app.route("/crop", methods=["GET", "POST"])
@login_required
def crop():
    if request.method == "POST":
        # 🌱 Soil Image First
        img = request.files["soil_image"]
        filepath = "temp_crop.jpg"
        img.save(filepath)

        soil_result = predict_soil(filepath)
        os.remove(filepath)

        # 🔥 Convert CNN label to encoder label
        mapped_soil = SOIL_LABEL_MAP.get(soil_result, soil_result)
        soil_encoded = soil_encoder.transform([mapped_soil])[0]

        # 🌾 Crop Inputs
        data = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"]),
            soil_encoded
        ]

        crop_result = crop_model.predict([data])[0]

        db.session.add(History(
            username=session["user"],
            prediction_type="Crop",
            result=f"Soil: {mapped_soil}, Crop: {crop_result}"
        ))
        db.session.commit()

        return render_template("result.html", result=f"Soil: {mapped_soil} | Crop: {crop_result}")

    return render_template("crop_form.html")

# ================= SOIL PREDICTION ONLY =================
@app.route("/soil", methods=["GET", "POST"])
@login_required
def soil():
    if request.method == "POST":
        img = request.files["soil_image"]
        filepath = "temp.jpg"
        img.save(filepath)

        result = predict_soil(filepath)
        os.remove(filepath)

        db.session.add(History(username=session["user"], prediction_type="Soil", result=result))
        db.session.commit()

        return render_template("result.html", result=result)

    return render_template("soil_form.html")

# ================= SOIL HEALTH METER =================
@app.route("/soil_health", methods=["GET", "POST"])
@login_required
def soil_health():
    if request.method == "POST":
        N = float(request.form.get("N", 0))
        P = float(request.form.get("P", 0))
        K = float(request.form.get("K", 0))
        ph = float(request.form.get("ph", 7.0))
        
        analysis = analyze_soil_health(N, P, K, ph)
        
        return render_template("soil_health_result.html", 
                             analysis=analysis,
                             N=N, P=P, K=K, ph=ph)
    
    return render_template("soil_health.html")

# ================= WEATHER FORECAST =================
@app.route("/weather", methods=["GET", "POST"])
@login_required
def weather():
    weather_data = None
    if request.method == "POST":
        city = request.form.get("city", "Hyderabad")
        weather_data = get_weather_data(city)
        if not weather_data:
            return render_template("weather.html", error="City not found or API error")
    
    return render_template("weather.html", weather_data=weather_data)

# ================= EXPENSE CALCULATOR =================
@app.route("/expense", methods=["GET", "POST"])
@login_required
def expense():
    if request.method == "POST":
        entry_type = request.form.get("entry_type")
        name = request.form.get("name")
        description = request.form.get("description")
        amount = float(request.form.get("amount", 0))
        date_str = request.form.get("date")

        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            date = datetime.utcnow().date()

        expense_entry = Expense(
            username=session["user"],
            entry_type=entry_type,
            name=name,
            description=description,
            amount=amount,
            date=date
        )

        db.session.add(expense_entry)
        db.session.commit()

        return redirect(url_for("expense_history"))

    # 👇 ADD THIS LINE
    today = datetime.now().strftime("%Y-%m-%d")

    return render_template("expense_form.html", today=today)

@app.route("/expense/history", methods=["GET", "POST"])
@login_required
def expense_history():
    # Get filter parameters
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    
    query = Expense.query.filter_by(username=session["user"])
    
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        query = query.filter(Expense.date >= start)
    
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        query = query.filter(Expense.date <= end)
    
    expenses = query.order_by(Expense.date.desc()).all()
    
    # Calculate totals
    total_expense = sum([e.amount for e in expenses if e.entry_type == 'expense'])
    total_income = sum([e.amount for e in expenses if e.entry_type == 'income'])
    balance = total_income - total_expense
    
    return render_template("expense_history.html",
                         expenses=expenses,
                         total_expense=total_expense,
                         total_income=total_income,
                         balance=balance,
                         start_date=start_date,
                         end_date=end_date)

# ================= GOVERNMENT SCHEMES =================
@app.route("/schemes")
@login_required
def schemes():
    return render_template("schemes.html", schemes=GOVERNMENT_SCHEMES)

# ================= GUIDE =================
@app.route("/guide")
def guide():
    return render_template("guide.html")

# ================= HISTORY =================
@app.route("/history")
@login_required
def history():
    records = History.query.filter_by(username=session["user"])\
        .order_by(History.timestamp.desc())\
        .all()
    return render_template("history.html", records=records)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("welcome"))

# ================= API ENDPOINTS =================
@app.route("/api/delete_expense/<int:expense_id>", methods=["DELETE"])
@login_required
def delete_expense(expense_id):
    expense = Expense.query.get(expense_id)
    if expense and expense.username == session["user"]:
        db.session.delete(expense)
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"success": False}), 403

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    