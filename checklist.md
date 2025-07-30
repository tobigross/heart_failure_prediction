# ✅ ML-Projekt Checkliste (für Industrie, Lebenslauf)

## 1. Validierung & Modellbewertung

- [x] Lade prediction & target_class in Power BI oder Pandas
- [X] Berechne:
  - [X] Accuracy
  - [X] Precision
  - [X] Recall
  - [X] F1-Score
- [X] Erstelle eine Konfusionsmatrix
- [X] Finde Fehlerfälle (prediction ≠ target_class)
  - [X] Visualisiere systematische Fehler (nach Alter, Cholesterin, etc.)

## 2. Dokumentation (README.md)

- [ ] Datenquelle & Beschreibung der Spalten
- [ ] Feature Engineering (One-Hot-Encoding, Umbenennungen)
- [ ] Modell: z. B. RandomForestClassifier + Parametereinstellungen
- [ ] API:
  - [ ] FastAPI-Inputstruktur
  - [ ] Beispiel-Request
- [ ] Power BI Report:
  - [ ] Beschreibung & Screenshots

## 3. Automatisierung

- [x] Erstelle ein ETL-Skript (`etl_predict.py`):
  - [x] Lade Daten
  - [x] Berechne Vorhersagen
  - [x] Speichere in SQL
- [ ] Zeitgesteuert ausführen:
  - [ ] Windows Task Scheduler / cron
- [ ] CI/CD mit GitHub Actions (optional):
  - [ ] API testen mit pytest
  - [ ] Formatierung prüfen (black, flake8)

## 4. Deployment

- [ ] Dockerfile für API
- [ ] API mit ngrok oder externem Server erreichbar machen
- [ ] API-Zugriff absichern (Token, Auth)

## 5. Weiterentwicklung

- [ ] Trainiere alternative Modelle (LogReg, XGBoost, etc.)
- [ ] Hyperparameter-Tuning mit GridSearch oder Optuna
- [ ] Feature Importance mit SHAP anzeigen
- [ ] Nutzerfeedback einbauen (Spalte „Feedback“)

## 6. Versionierung & Teamfähigkeit

- [ ] Modell & Daten versionieren mit DVC
- [ ] Projektstruktur einhalten:
  - `api/`, `ml/`, `sql/`, `data/`, `bi/`, `docs/`
- [ ] `.gitignore` & `.env` verwenden
- [ ] Tests:
  - [ ] FastAPI-Tests
  - [ ] Pipeline-Tests
