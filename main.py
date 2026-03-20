from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from google import genai 
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv  # Tambahan: Untuk membaca file .env

# --- LANGKAH 1: KONFIGURASI KEAMANAN ---
load_dotenv()  # Muat file .env
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Inisialisasi Client Gemini secara aman
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("⚠️ Peringatan: GOOGLE_API_KEY tidak ditemukan di .env!")

# --- LANGKAH 2: INISIALISASI DATABASE ---
def init_db():
    conn = sqlite3.connect('interview_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (timestamp TEXT, emotion TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db() 

app = FastAPI()

# --- LANGKAH 3: MIDDLEWARE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LANGKAH 4: LOAD MODEL AI ---
emotion_model = tf.keras.models.load_model('models/interview_ai_model_v2.keras')
emotions_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.get("/")
def home():
    return {"message": "API Interview-AI Modern Aktif!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Baca File Gambar
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # 2. Preprocessing untuk Model AI
    img_resized = cv2.resize(img, (48, 48))
    img_normalized = img_resized / 255.0
    img_final = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)
    
    # 3. Prediksi Emosi
    prediction = emotion_model.predict(img_final)
    max_index = int(np.argmax(prediction))
    predicted_emotion = emotions_list[max_index]
    confidence_value = float(np.max(prediction)) # Simpan skor keyakinan

    # 4. Panggil Gemini (Generative AI)
    ai_suggestion = ""
    prompt = f"Seorang kandidat interview merasa {predicted_emotion}. Berikan 1 kalimat motivasi yang sangat pendek, bersemangat, dan empatik untuknya."
    
    try:
        if client:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt
            )
            ai_suggestion = response.text.strip()
        else:
            raise Exception("Client Gemini tidak terinisialisasi")
            
    except Exception as e:
        print(f"DEBUG LOG: Menggunakan Fallback karena: {e}")
        # Logika Fallback jika API Limit/Error
        fallback_quotes = {
            "Fear": "Tarik napas dalam, rasa gugup adalah tanda kamu peduli. Kamu pasti bisa!",
            "Happy": "Pertahankan energi positif ini, senyummu adalah aset terbaikmu!",
            "Sad": "Jangan biarkan satu keraguan menghapus persiapan matangmu. Bangkit lagi!",
            "Angry": "Tenangkan pikiran, fokus pada solusi, bukan pada emosi saat ini.",
            "Neutral": "Kamu terlihat tenang, gunakan ketenangan ini untuk menjawab dengan lugas.",
            "Surprise": "Tetap fokus, jadikan kejutan ini sebagai tantangan yang menarik!",
            "Disgust": "Alihkan fokusmu ke hal profesional, kamu punya kendali penuh atas dirimu."
        }
        ai_suggestion = fallback_quotes.get(predicted_emotion, "Tetap tenang dan lakukan yang terbaik!")

    # 5. Simpan ke Database (History)
    try:
        conn = sqlite3.connect('interview_history.db')
        c = conn.cursor()
        c.execute("INSERT INTO history VALUES (?, ?, ?)", 
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), predicted_emotion, confidence_value))
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"Gagal simpan ke Database: {db_err}")

    # 6. Kirim Hasil ke Frontend
    return {
        "emotion": predicted_emotion,
        "confidence": confidence_value,
        "motivation_quote": ai_suggestion 
    }