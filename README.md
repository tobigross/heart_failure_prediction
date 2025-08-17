# Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green?logo=fastapi) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13-blue?logo=postgresql) ![PowerBI](https://img.shields.io/badge/PowerBI-Desktop-yellow?logo=microsoft-power-bi)

An end-to-end machine learning application that predicts heart disease based on patient health data. This project contains a complete ML pipeline from data ingestion to model deployment.

This is a toy/educational project only. Do not use for real medical decisions. Always consult a healthcare professional for medical advice.
---

## Key Features

- **ETL Pipeline:** Automated data loading, processing and saving
- **Multi-Model Comparison:** Four ML models trained and evaluated (PyTorch NN, Random Forest, 2 Gradient Boosting)
- **API:** FastAPI backend
- **Database Integration:** PostgreSQL for prediction storage and retrieval
- **Dashboard:** Power BI visualization for model insights
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
- Git for version control
**Tools & Productivity:**
- Git for version control
- LLMs (ChatGPT, Claude, Gemini, Cursor) — used for rapid learning of new technologies (frontend, APIs, ETL pipelines, PostgreSQL) and informed decision-making on system design and best practices
---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| PyTorch Neural Network | 0.89 | 0.89 | 0.89 | 0.89 | 0.87 |
| sklearn Random Forest | 0.88 | 0.88 | 0.88 | 0.88 | 0.937 |
| sklearn Gradient Boosting (Tuned) | 0.886 | 0.886 | 0.886 | 0.886 | 0.941 |
| sklearn Gradient Boosting (Untuned) | 0.880 | 0.882 | 0.880 | 0.878 | 0.938 |

<details>
<summary>Detailed Classification Reports</summary>


### PyTorch Neural Network

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.89    | 0.83   | 0.86     | 112     |
| 1 (Disease)    | 0.89    | 0.93   | 0.91     | 164     |
| **Accuracy**   |         |        | 0.89 | 276     |
| **Macro Avg**  | 0.89    | 0.88   | 0.89     | 276     |
| **Weighted Avg** | 0.89  | 0.89   | 0.89     | 276     |

**ROC-AUC:** 0.874

### Random Forest

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.82    | 0.88   | 0.85     | 112     |
| 1 (Disease)    | 0.92    | 0.87   | 0.89     | 164     |
| **Accuracy**   |         |        | 0.88 | 276     |
| **Macro Avg**  | 0.87    | 0.88   | 0.87     | 276     |
| **Weighted Avg** | 0.88  | 0.88   | 0.88     | 276     |

**ROC-AUC:** 0.937

### Gradient Boosting (Untuned)

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.840   | 0.883  | 0.861    | 77      |
| 1 (Disease)    | 0.913   | 0.879  | 0.895    | 107     |
| **Accuracy**   |         |        | 0.880 | 184     |
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
| **Accuracy**   |         |        | 0.886 | 184     |
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

---

## Dataset

**Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by Soriano, F. (2021)

**Key Features:**
- **918 records** with 11 features
- **Target variable:** Heart disease presence (binary classification)
- **No preprocessing:** No missing values or similar. Everything is ready to go.

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

One-hot encoding is used for all models.

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


## Usage

### Making Predictions via API

```python
import requests

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Patient data
data = {
    "Age": 54,
    "Sex": 1,  
    "ChestPainType": 0, 
    "RestingBP": 108,
    "Cholesterol": 267,
    "FastingBS": 0,  
    "RestingECG": 1,  
    "MaxHR": 167,
    "ExerciseAngina": 0,  
    "Oldpeak": 0,
    "ST_Slope": 2  
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

# Print results
for model, prediction in result.items():
    risk = "Heart Disease" if prediction else "Normal"
    print(f"{model}: {risk}")
```

### Web Interface
1. Start API
2. Open frontend/index.html
3. Input patient data through the form
4. View prediction results

---

## Project Structure


```
heart_failure_prediction
├─ api
│  └─ main.py
├─ frontend
│  └─ index.html
├─ images
│  ├─ diagramm.png
│  ├─ report.png
│  └─ tuned_feature_importance.jpeg
├─ ml
│  ├─ get_ROC.py
│  ├─ gradient_boost_default.pkl
│  ├─ gradient_boost_tuned.pkl
│  ├─ model.py
│  ├─ model_gradientboost.py
│  ├─ model_pytorch.pt
│  ├─ model_sklearn.pkl
│  └─ training_columns.txt
├─ notebooks
│  ├─ DataExploration.ipynb
│  ├─ test_model.ipynb
│  └─ UpdateSQL.ipynb
├─ tests
│   └─ test_api.py
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ etl_mini_2.py


```
---

## Key Insights

- NN has worst ROC-AUC
- The tuned Gradient boost model slightly better then default

---
## Analytics Dashboard
The complete dataset is used in PowerBi.
In this report the Random Forest performs the best.
The large discrepancy is probably due to overfitting.
This could be changed with more data or regularization.

Power BI dashboard:
![Dashboard Preview](./images/report.png)

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Tobias Groß**  
tobigross3334@gmail.com

---

## Acknowledgments

- Dataset provided by Soriano, F. via Kaggle
