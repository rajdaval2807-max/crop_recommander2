🌾 Agro-AIAB — AI Powered Agriculture Assistant
An end-to-end AI-powered web platform that helps farmers and agriculture enthusiasts make smarter decisions using Machine Learning and real-time data.
Built using Flask + AI/ML + Weather APIs, this project predicts crops, analyzes soil health, forecasts weather, and even tracks farm expenses — all in one dashboard.
________________________________________
🚀 Features
•	🌱 Crop Recommendation System (ML model)
•	🧪 Soil Type Detection using CNN image model
•	📊 Soil Health Analyzer (NPK + pH insights)
•	🌦️ 5-Day Weather Forecast with warnings
•	💰 Farm Expense Tracker (income vs expenses)
•	📜 Prediction History Tracking
•	🏛️ Government Schemes Explorer
•	🔐 Secure login/signup system
________________________________________
🧠 Tech Stack
Layer	Technology
Backend	Flask (Python)
ML Models	TensorFlow, scikit-learn
Database	SQLite (SQLAlchemy ORM)
APIs	OpenWeatherMap API
Frontend	HTML, CSS, JavaScript
Deployment	Render / Railway
________________________________________

📦 Project Structure
agro-aiab/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/              # Auto-downloaded ML models
├── datasets/            # Sample datasets
├── templates/           # HTML templates
├── static/              # CSS, JS, images
└── instance/            # SQLite database (ignored in Git)
________________________________________
🤖 Machine Learning Models
Due to GitHub file size limits, trained models are stored on Google Drive and downloaded automatically on first run.
Model	Purpose
crop_model.pkl	Crop recommendation
soil_model.h5	CNN soil classification
soil_encoder.pkl	Soil label encoding
The application will automatically download these models into the models/ folder if they are missing.
________________________________________
📊 Datasets
This repository includes sample datasets for demonstration purposes.
Full datasets are hosted externally to keep the repository lightweight.
You can:
•	Use sample datasets for testing
•	Replace with your own datasets
•	Download full datasets from external links (optional)
________________________________________
⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-username/agro-aiab.git
cd agro-aiab
2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\\Scripts\\activate      # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Set environment variables
Create a .env file or export manually:
WEATHER_API_KEY=your_openweather_api_key
________________________________________
5️⃣ Run the app
python app.py
App will be available at:
http://127.0.0.1:5000
On first run:
•	ML models will auto-download
•	Database tables will auto-create
________________________________________
🌍 Deployment
This project can be deployed easily on:
•	Render (recommended)
•	Railway
•	Docker
Deployment Notes
•	Use a persistent database (Postgres) for production
•	Set environment variables in platform dashboard
________________________________________
🔒 Authentication
•	Secure password hashing
•	Session-based authentication
•	User-specific prediction history
________________________________________
📈 Future Improvements
•	PostgreSQL production database
•	Mobile responsive UI
•	Crop disease detection
•	Multi-language support
•	AI chatbot assistant
________________________________________
🤝 Contributing
Contributions are welcome!
If you'd like to improve this project:
1.	Fork the repo
2.	Create a feature branch
3.	Submit a pull request
________________________________________
📜 License
This project is licensed under the MIT License.
________________________________________
👨💻 Author
Raj
B.Sc IT Student | Aspiring AI Developer
If you found this project helpful, consider ⭐ starring the repo!

