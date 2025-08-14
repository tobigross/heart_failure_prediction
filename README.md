# Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green?logo=fastapi) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13-blue?logo=postgresql) ![PowerBI](https://img.shields.io/badge/PowerBI-Desktop-yellow?logo=microsoft-power-bi)

An end-to-end machine learning application that predicts heart disease likelihood based on patient health data, achieving **89% accuracy** with a PyTorch neural network. This project demonstrates the complete ML pipeline from data ingestion to model deployment.

---

## Key Features

- **Complete ETL Pipeline:** Automated data ingestion, cleaning, and preprocessing
- **Multi-Model Comparison:** Four ML models trained and evaluated (PyTorch NN, Random Forest, Gradient Boosting)
- **Production-Ready API:** FastAPI backend with comprehensive error handling
- **Database Integration:** PostgreSQL for prediction storage and retrieval
- **Interactive Dashboard:** Power BI visualization for model insights
- **Web Interface:** User-friendly HTML frontend for predictions

---

## Tech Stack

**Backend & ML:**
- Python 3.10, scikit-learn, PyTorch
- FastAPI for REST API
- PostgreSQL for data persistence

**Frontend & Visualization:**
- HTML/CSS/JavaScript
- Power BI for analytics dashboard

**Tools & Deployment:**
- Docker for containerization
- Git for version control

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **PyTorch Neural Network** | **0.89** | **0.89** | **0.89** | **0.89** | **0.89** |
| Random Forest | 0.88 | 0.88 | 0.88 | 0.88 | 0.94 |
| Gradient Boosting (Tuned) | 0.886 | 0.886 | 0.886 | 0.886 | 0.941 |
| Gradient Boosting (Untuned) | 0.880 | 0.882 | 0.880 | 0.878 | 0.938 |

<details>
<summary>Detailed Classification Reports</summary>

### PyTorch Neural Network

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.89    | 0.83   | 0.86     | 112     |
| 1 (Disease)    | 0.89    | 0.93   | 0.91     | 164     |
| **Accuracy**   |         |        | **0.89** | 276     |
| **Macro Avg**  | 0.89    | 0.88   | 0.89     | 276     |
| **Weighted Avg** | 0.89  | 0.89   | 0.89     | 276     |

### Random Forest

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.82    | 0.88   | 0.85     | 112     |
| 1 (Disease)    | 0.92    | 0.87   | 0.89     | 164     |
| **Accuracy**   |         |        | **0.88** | 276     |
| **Macro Avg**  | 0.87    | 0.88   | 0.87     | 276     |
| **Weighted Avg** | 0.88  | 0.88   | 0.88     | 276     |

### Gradient Boosting (Untuned)

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.840   | 0.883  | 0.861    | 77      |
| 1 (Disease)    | 0.913   | 0.879  | 0.895    | 107     |
| **Accuracy**   |         |        | **0.880** | 184     |
| **Macro Avg**  | 0.876   | 0.881  | 0.878    | 184     |
| **Weighted Avg** | 0.882 | 0.880  | 0.881    | 184     |

**ROC-AUC:** 0.938

**Confusion Matrix:**
|           | Predicted 0 | Predicted 1 |
|-----------|-------------|-------------|
| **Actual 0** | 68       | 9           |
| **Actual 1** | 13       | 94          |

### Gradient Boosting (Tuned)

**Best Parameters:** `{'n_estimators': 199, 'learning_rate': 0.012, 'max_depth': 3, 'subsample': 0.54}`

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.859   | 0.870  | 0.865    | 77      |
| 1 (Disease)    | 0.906   | 0.897  | 0.901    | 107     |
| **Accuracy**   |         |        | **0.886** | 184     |
| **Macro Avg**  | 0.882   | 0.884  | 0.883    | 184     |
| **Weighted Avg** | 0.886 | 0.886  | 0.886    | 184     |

**ROC-AUC:** 0.941

**Confusion Matrix:**
|           | Predicted 0 | Predicted 1 |
|-----------|-------------|-------------|
| **Actual 0** | 67       | 10          |
| **Actual 1** | 11       | 96          |

**Feature importance**

![Feature importance](./images/tuned_feature_importance.jpeg)

</details>

---

## Architecture

![Project Architecture](./images/diagramm.png)

The system follows a modular architecture with clear separation between data processing, model training, API services, and user interfaces.

---

## Dataset

**Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by Soriano, F. (2021)

**Key Features:**
- **918 records** with 11 clinical features
- **Target variable:** Heart disease presence (binary classification)
- **Features include:** Age, chest pain type, blood pressure, cholesterol, ECG results, exercise capacity

<details>
<summary>Detailed Feature Descriptions</summary>

| Feature | Description | Values/Units |
|---------|-------------|--------------|
| Age | Patient age | Years (28-77) |
| Sex | Patient gender | M (Male), F (Female) |
| ChestPainType | Type of chest pain | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure | mm Hg (0-200) |
| Cholesterol | Serum cholesterol | mg/dL (0-603) |
| FastingBS | Fasting blood sugar > 120 mg/dL | 0, 1 |
| RestingECG | Resting ECG results | Normal, ST, LVH |
| MaxHR | Maximum heart rate | BPM (60-202) |
| ExerciseAngina | Exercise-induced angina | Y (Yes), N (No) |
| Oldpeak | ST depression | Numeric (-2.6 to 6.2) |
| ST_Slope | Peak exercise ST segment slope | Up, Flat, Down |

</details>

---

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 13+
- Git

### Installation

Follow these steps to set up and run the project:
<details>
<summary>Installation steps</summary>
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tobigross/heart_failure_prediction.git
    cd heart_failure_prediction 
    ```
    ---

2.  **Set Up Conda Environment:**
    If you don't have Conda installed, please download and install Miniconda or Anaconda first.

    * **Create a new conda environment:**
        ```bash
        conda create --name heart_failure_prediction python=3.10
        ```

    * **Activate the environment:**
        ```bash
        conda activate heart_failure_prediction
        ```

    * **Install project dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
    ---

3.  **Set Up Database:**
    Ensure you have PostgreSQL or your chosen database system installed and running.

    * **Create the database:**
        ```bash
        createdb heart_disease_db
        ```
    * **Run database setup script:**
        ```bash
        python setup_database.py
        ```
    ---

4.  **Configure Environment Variables:**
    This project uses environment variables for sensitive information like database credentials.

    * **Copy the example file:**
        ```bash
        cp .env.example .env
        ```
    * **Edit `.env`:** Open the newly created `.env` file in a text editor and update it with your actual database credentials and any other necessary configurations.
        *(e.g., `DATABASE_URL=postgresql://user:password@host:port/database_name`)*
    ---

5.  **Run the ETL Process:**

    * **Execute the ETL script:**
        ```bash
        python etl_mini_2.py
        ```
</details>

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application at http://localhost:8000
```

---

## Usage

### Making Predictions via API

```python
import requests

# Sample prediction request
data = {
    "age": 54,
    "sex": "M",
    "chest_pain_type": "ATA",
    "resting_bp": 120,
    "cholesterol": 188,
    "fasting_bs": 0,
    "resting_ecg": "Normal",
    "max_hr": 113,
    "exercise_angina": "N",
    "oldpeak": 1.4,
    "st_slope": "Flat"
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
print(f"Heart Disease Risk: {prediction['risk_level']}")
```

### Web Interface
1. Navigate to `http://localhost:8000/dashboard`
2. Input patient data through the form
3. View prediction results and risk assessment

---

## Project Structure

```
heart-disease-prediction/
├── data/                   # Dataset and processed data
├── models/                 # Trained model files
├── src/
│   ├── data_processing/    # ETL pipeline
│   ├── models/            # ML model implementations
│   ├── api/               # FastAPI application
│   └── utils/             # Helper functions
├── frontend/              # HTML/CSS/JS files
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit and integration tests
├── docker-compose.yml     # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Key Insights

- **Age and cholesterol** are the strongest predictors of heart disease
- **Exercise-induced angina** shows high correlation with positive cases
- **ST slope patterns** provide crucial diagnostic information
- Model achieves **89% accuracy** on unseen data

---
## Analytics Dashboard

Power BI dashboard providing comprehensive analysis of:

- **Model Performance Metrics** - ROC curves, confusion matrices, feature importance
- **Dataset Exploration** - Distribution analysis, correlation heatmaps, demographic breakdowns  
- **Prediction Analytics** - Real-time prediction tracking, risk assessment visualizations
- **Clinical Insights** - Patient cohort analysis, risk factor trends

![Dashboard Preview](./images/powerbi-dashboard.png)

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Tobias Groß**  


---

## Acknowledgments

- Dataset provided by Soriano, F. via Kaggle