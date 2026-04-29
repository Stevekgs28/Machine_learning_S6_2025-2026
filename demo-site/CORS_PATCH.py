# Patch CORS pour code/api.py
# ============================
# 
# Le front Vite tourne sur http://localhost:5173 ; sans CORS le navigateur
# bloquera tous les appels POST /predict. Ajouter ces 12 lignes en haut de
# code/api.py juste après la ligne `app = FastAPI(...)`.
#
# 1. Ajouter l'import en haut du fichier (avec les autres imports) :

from fastapi.middleware.cors import CORSMiddleware

# 2. Juste après le `app = FastAPI(...)`, ajouter :

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://127.0.0.1:5173",
        # Ajouter ici l'URL de prod si le site est déployé (Vercel/Netlify/Pages)
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# C'est tout. Relance uvicorn et le front pourra appeler /predict.
