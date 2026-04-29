# steve-ml-cybersec — demo site

Site de démo pour le projet ML MOD10 Winter 2026. Sert à présenter le pipeline,
les modèles, le diagnostic statistique et le Pivot B en une page interactive.
Le formulaire **Predict** appelle l'API FastAPI réelle (`code/api.py`).

## Stack

- **Vite 5 + React 18** — bundle léger, page unique, déploiement statique
- **TailwindCSS 3** — styling, thème dark "terminal-ML" sur-mesure
- **Recharts** — bar charts pour la comparaison modèles + Pivot B
- **Polices** : JetBrains Mono (terminal) + Space Grotesk (display)

## Démarrage rapide

```bash
# 1. Installer les deps
npm install

# 2. Lancer le dev server (hot reload)
npm run dev
# → http://localhost:5173

# 3. Build de production
npm run build
# → dist/ contient un site statique déployable n'importe où
```

## API — préalable obligatoire

Le playground appelle `POST /predict` sur l'API FastAPI. Avant la démo :

```bash
# Depuis la racine du projet (Machine_learning_S6_2025-2026/)
cd code
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# OU via Docker
docker build -t steve-ml-api .
docker run -p 8000:8000 steve-ml-api
```

### Patch CORS à appliquer une fois

Sans CORS, le navigateur bloque tous les appels du front. Ajouter ces lignes
dans `code/api.py` (voir `CORS_PATCH.py` pour le détail) :

```python
from fastapi.middleware.cors import CORSMiddleware

# juste après : app = FastAPI(...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```

## Override de l'URL d'API

Par défaut le front cherche l'API sur `http://localhost:8000`. Pour pointer
ailleurs (laptop d'un coéquipier, ngrok, IP locale dans une démo en salle) :

```
http://localhost:5173/?api=https://abcd.ngrok.io
```

Ou dans la console navigateur :

```js
localStorage.setItem('api_base', 'http://192.168.1.42:8000')
location.reload()
```

Le badge en haut à droite affiche en temps réel l'état de l'API :
- 🟢 **vert** : API + modèle chargé → prédictions OK
- 🟡 **ambre** : API joignable mais modèle pas chargé
- 🔴 **rouge** : API offline

## Structure

```
demo-site/
├── public/
│   ├── favicon.svg
│   └── figures/                # PNG copiés de graphs/ + racine du projet
├── src/
│   ├── App.jsx                 # nav + layout + observation des sections
│   ├── main.jsx
│   ├── index.css               # tailwind + thème terminal
│   ├── data/models.js          # 6 modèles, Pivot B, options form, rubric
│   ├── lib/api.js              # fetch /predict + /health
│   └── sections/
│       ├── Hero.jsx            # pitch + 3 stats killer + terminal animé
│       ├── Playground.jsx      # formulaire 9 champs + appel API live
│       ├── ModelComparison.jsx # tableau + barchart Recharts + figures
│       ├── Diagnostic.jsx      # 4 hypothèses + 5 tests + 7 figures EDA/SHAP
│       ├── PivotB.jsx          # la slide-killer
│       └── About.jsx           # équipe + couverture rubrique + stack
├── index.html
├── tailwind.config.js
├── postcss.config.js
├── vite.config.js
└── package.json
```

## Sections du site

| # | Section | Rôle dans la défense |
|---|---|---|
| 00 | **Pitch** | Phrase pivot + 3 chiffres-killer (KS p=0.804, AUC 0.50, Pivot B F1 0.97) |
| 01 | **Predict** | Démo live `/predict` — formulaire 9 champs + sélecteur modèle |
| 02 | **Models** | Tableau des 6 modèles + barchart F1/AUC + ROC curves + confusion matrices |
| 03 | **Diagnostic** | 4 hypothèses, tableau des 5 tests stats, 7 figures EDA + SHAP |
| 04 | **Pivot B** | Figure 08 + chart comparatif + tableau Steve vs PhishingWebsites |
| 05 | **About** | Équipe + couverture rubrique 8/8 + stack |

## Personnalisation

- **Chiffres des modèles** : `src/data/models.js` (constants `MODELS`, `PIVOT_B`, `DIAGNOSTIC_TESTS`)
- **Options du formulaire** : `src/data/models.js` (`FORM_OPTIONS`, `FORM_DEFAULTS`)
- **Couleurs / typo** : `tailwind.config.js` + `src/index.css`
- **Sections de la nav** : `src/App.jsx` (constante `NAV`)

## Déploiement

`dist/` est un site 100% statique. Déployable sur :
- **Vercel / Netlify / GitHub Pages** : drop du dossier `dist/`
- Mettre à jour le `allow_origins` du CORS avec l'URL de prod
- L'API doit être joignable publiquement (ngrok pour la démo, ou un déploiement
  Cloud Run / Fly.io / Railway pour quelque chose de pérenne)

## Pour la soutenance — checklist Steve

- [ ] `uvicorn api:app --port 8000` lancé sur le laptop avant l'oral
- [ ] CORS appliqué sur `code/api.py`
- [ ] `npm run dev` ou `npm run build && npm run preview` lancé
- [ ] Vérifier le badge API vert en haut à droite
- [ ] Prévoir 1 prédiction d'exemple à montrer (Random ⚄ ou inputs précis)
- [ ] Avoir l'URL `localhost:5173` ouverte sur la slide de transition
