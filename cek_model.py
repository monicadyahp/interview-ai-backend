import google.generativeai as genai

# Ganti dengan API Key-mu
genai.configure(api_key="ISI_API_KEY_KAMU")

print("--- Daftar Model yang Tersedia untuk Kamu ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Nama Model: {m.name}")
except Exception as e:
    print(f"Error saat cek: {e}")