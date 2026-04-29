// Données issues du tableau de résultats du rapport et des notes README/argumentation-defense.
// Source : Note-Obisidian/argumentation-defense.md (section Résultats)

export const MODELS = [
  {
    id: 'baseline',
    name: 'Dummy (most_frequent)',
    short: 'Baseline',
    family: 'baseline',
    train_f1: 0.428,
    test_f1: 0.429,
    accuracy: 0.750,
    roc_auc: 0.500,
    gap: -0.000,
    notes: 'Toujours prédire la classe majoritaire. Sert de plancher.',
    rocFigure: null,
    cmFigure: '/figures/confusion_matrix_baseline_most_frequent.png',
  },
  {
    id: 'logistic_regression',
    name: 'Logistic Regression',
    short: 'LogReg',
    family: 'linear',
    train_f1: 0.525,
    test_f1: 0.477,
    accuracy: 0.526,
    roc_auc: 0.498,
    gap: 0.047,
    notes: 'Modèle linéaire, class_weight=balanced. AUC ≈ 0.5 → pas de séparation linéaire.',
    rocFigure: '/figures/roc_logistic_regression.png',
    cmFigure: '/figures/confusion_matrix_logistic_regression.png',
  },
  {
    id: 'decision_tree',
    name: 'Decision Tree',
    short: 'DT',
    family: 'tree',
    train_f1: 0.651,
    test_f1: 0.483,
    accuracy: 0.509,
    roc_auc: 0.524,
    gap: 0.168,
    notes: 'Arbre simple, profondeur tunée par GridSearchCV. Léger overfit.',
    rocFigure: '/figures/roc_decision_tree.png',
    cmFigure: '/figures/confusion_matrix_decision_tree.png',
  },
  {
    id: 'random_forest',
    name: 'Random Forest',
    short: 'RF',
    family: 'ensemble',
    train_f1: 0.910,
    test_f1: 0.491,
    accuracy: 0.703,
    roc_auc: 0.505,
    gap: 0.418,
    notes: 'Champion sérialisé. Train F1=0.91 prouve que la capacité du modèle n\'est pas le bottleneck — c\'est la donnée.',
    rocFigure: '/figures/roc_random_forest.png',
    cmFigure: '/figures/confusion_matrix_random_forest.png',
    isChampion: true,
  },
  {
    id: 'voting_soft',
    name: 'Voting (soft, LR + DT)',
    short: 'Voting',
    family: 'ensemble',
    train_f1: 0.649,
    test_f1: 0.481,
    accuracy: 0.508,
    roc_auc: 0.521,
    gap: 0.168,
    notes: 'Moyenne pondérée des probas LR + DT.',
    rocFigure: '/figures/roc_voting_soft.png',
    cmFigure: '/figures/confusion_matrix_voting_soft.png',
  },
  {
    id: 'stacking',
    name: 'Stacking (LR+DT → meta LR)',
    short: 'Stacking',
    family: 'ensemble',
    train_f1: 0.488,
    test_f1: 0.479,
    accuracy: 0.510,
    roc_auc: 0.514,
    gap: 0.009,
    notes: 'Meilleur écart train/test (+0.009) — le plus honnête statistiquement.',
    rocFigure: '/figures/roc_stacking.png',
    cmFigure: '/figures/confusion_matrix_stacking.png',
  },
]

// Pivot B : même pipeline sur PhishingWebsites (OpenML 4534)
export const PIVOT_B = [
  { model: 'Baseline', steve_f1: 0.4286, phishing_f1: 0.3577, phishing_auc: 0.500 },
  { model: 'LogReg', steve_f1: 0.4775, phishing_f1: 0.9258, phishing_auc: 0.9784 },
  { model: 'Decision Tree', steve_f1: 0.4828, phishing_f1: 0.9617, phishing_auc: 0.9780 },
  { model: 'Random Forest', steve_f1: 0.4914, phishing_f1: 0.9697, phishing_auc: 0.9973 },
]

// Diagnostic statistique
export const DIAGNOSTIC_TESTS = [
  {
    test: 'Kolmogorov–Smirnov',
    target: 'Financial Loss vs Uniform[0.5, 99.99]',
    statistic: 'D = 0.012',
    pvalue: '0.804',
    verdict: 'Distribution uniforme aléatoire non rejetable',
    severity: 'critical',
  },
  {
    test: 'Mutual Information',
    target: 'Year, # Affected Users, Resolution Time',
    statistic: 'MI ≤ 0.006',
    pvalue: '— (seuil < 0.02)',
    verdict: 'Aucune information mutuelle exploitable',
    severity: 'critical',
  },
  {
    test: 'Cramér\'s V + chi²',
    target: 'Country, Attack Type, Industry, …',
    statistic: 'V max = 0.058',
    pvalue: '≥ 0.34',
    verdict: 'Indépendance non rejetée à α=0.05',
    severity: 'critical',
  },
  {
    test: 'Régression directe',
    target: 'Linear + RF sur Financial Loss continu',
    statistic: 'R² test = -0.017 / -0.034',
    pvalue: '—',
    verdict: 'Pire que prédire la moyenne',
    severity: 'critical',
  },
  {
    test: 'SHAP TreeExplainer',
    target: 'Random Forest (44 features OneHot)',
    statistic: 'mean |SHAP| max = 0.0171',
    pvalue: '—',
    verdict: 'Importance diluée, aucune feature dominante',
    severity: 'warning',
  },
]

// Options des dropdowns extraites du dataset (preprocess.py + dataset.csv)
export const FORM_OPTIONS = {
  Country: ['Australia', 'Brazil', 'China', 'France', 'Germany', 'India', 'Japan', 'Russia', 'UK', 'USA'],
  'Attack Type': ['DDoS', 'Malware', 'Man-in-the-Middle', 'Phishing', 'Ransomware', 'SQL Injection'],
  'Target Industry': ['Banking', 'Education', 'Government', 'Healthcare', 'IT', 'Retail', 'Telecommunications'],
  'Attack Source': ['Hacker Group', 'Insider', 'Nation-state', 'Unknown'],
  'Security Vulnerability Type': ['Social Engineering', 'Unpatched Software', 'Weak Passwords', 'Zero-day'],
  'Defense Mechanism Used': ['AI-based Detection', 'Antivirus', 'Encryption', 'Firewall', 'VPN'],
}

export const FORM_DEFAULTS = {
  Country: 'Germany',
  Year: 2023,
  'Attack Type': 'Ransomware',
  'Target Industry': 'Banking',
  'Number of Affected Users': 150000,
  'Attack Source': 'Nation-state',
  'Security Vulnerability Type': 'Zero-day',
  'Defense Mechanism Used': 'Encryption',
  'Incident Resolution Time (in Hours)': 48,
}

export const TEAM = [
  { name: 'Noa ISABEY' },
  { name: 'Léo PROSPER' },
  { name: 'Ashkan MIRI' },
  { name: 'Steve KINGSLEY ARULNEASATHAS' },
]

export const RUBRIC = [
  { component: 'Preprocessing & EDA', weight: 20, status: 'delivered', artifact: 'preprocess.py + EDA figures' },
  { component: 'Model Training', weight: 15, status: 'delivered', artifact: 'LR + DT + RF + GridSearchCV' },
  { component: 'Ensembles', weight: 10, status: 'delivered', artifact: 'Voting (soft) + Stacking' },
  { component: 'Evaluation Metrics', weight: 15, status: 'delivered', artifact: 'F1 macro, accuracy, ROC-AUC' },
  { component: 'Interpretability', weight: 10, status: 'delivered', artifact: 'SHAP TreeExplainer' },
  { component: 'Docker + REST API', weight: 10, status: 'delivered', artifact: 'FastAPI + Dockerfile' },
  { component: 'MLflow', weight: 10, status: 'delivered', artifact: '2 expériences, 10 runs' },
  { component: 'Presentation', weight: 10, status: 'delivered', artifact: 'Soutenance orale + ce site' },
]
