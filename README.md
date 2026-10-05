# Customer Churn Prediction using Machine Learning & Deep Learning

An end-to-end customer churn prediction project that compares multiple Machine Learning models with a Neural Network and provides an interactive Gradio dashboard for real-time predictions.

## Project Overview

Customer churn is a major business problem: identifying customers who are likely to leave a service can help companies take preventive actions such as targeted offers, customer support, or retention campaigns.

This project builds a binary classification system that predicts whether a customer is likely to **Churn** or **Not Churn** based on customer demographic, contract, service, and billing information.

The project compares four different models:

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Artificial Neural Network (ANN)** using PyTorch

An interactive **Gradio** interface is also included so users can enter customer information and receive a churn prediction and probability.

---

## Features

- Data preparation and preprocessing
- Missing-value handling
- Categorical feature encoding using `LabelEncoder`
- Feature scaling using `StandardScaler`
- Exploratory Data Analysis (EDA)
- Data visualization
- Multiple Machine Learning models
- Deep Learning Neural Network using PyTorch
- Model comparison using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Confusion matrices
- Neural Network training loss and accuracy curves
- New-customer prediction function
- Interactive Gradio prediction dashboard

---

## Dataset

### Important Note

The current notebook generates a **synthetic dataset** inside the code using NumPy. It does **not** load an external real-world customer churn CSV.

The generated dataset contains **1,000 customer records** with the following features:

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Age` | Customer age |
| `Tenure` | Number of months with the company |
| `MonthlyCharges` | Monthly customer charges |
| `TotalCharges` | Total customer charges |
| `Gender` | Customer gender |
| `ContractType` | Customer contract type |
| `TechSupport` | Whether the customer has technical support |
| `InternetService` | Type of internet service |
| `Churn` | Target variable: 0 = No Churn, 1 = Churn |

The notebook also introduces approximately **2% missing values** in `TotalCharges` and fills them using the median.

> For a production-quality or academic real-world ML project, replace the synthetic data generation step with a genuine customer churn dataset and document its source.

---

## Project Workflow

```text
Data Generation
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Categorical Encoding
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Exploratory Data Analysis
      ↓
Machine Learning Models
      ├── KNN
      ├── Random Forest
      └── SVM
      ↓
Deep Learning
      └── Neural Network
      ↓
Model Evaluation
      ├── Accuracy
      ├── Precision
      ├── Recall
      └── F1-Score
      ↓
Model Comparison
      ↓
Customer Prediction
      ↓
Gradio Interactive Dashboard
```

---

## Data Preprocessing

### 1. Missing Values

Missing values are introduced in `TotalCharges` and replaced with the median:

```python
df['TotalCharges'] = df['TotalCharges'].fillna(
    df['TotalCharges'].median()
)
```

### 2. Categorical Encoding

Categorical columns are transformed into numerical values using `LabelEncoder`:

- Gender
- ContractType
- TechSupport
- InternetService

### 3. Train/Test Split

The data is divided into:

- **80% training data**
- **20% testing data**

Stratified splitting is used to preserve the class distribution.

### 4. Feature Scaling

`StandardScaler` is applied to the features before training the models.

---

## Exploratory Data Analysis

The notebook includes visualizations for:

- Churn class distribution
- Age distribution by churn status
- Monthly charges by churn status

These visualizations provide an initial understanding of the relationship between customer characteristics and churn.

---

## Machine Learning Models

### K-Nearest Neighbors (KNN)

KNN predicts a customer's class based on nearby training observations.

Configuration:

```python
KNeighborsClassifier(n_neighbors=5)
```

### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees.

Configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Support Vector Machine (SVM)

SVM attempts to find an effective decision boundary between the two classes.

Configuration:

```python
SVC(
    probability=True,
    random_state=42
)
```

---

## Deep Learning Model

The project also implements an Artificial Neural Network using **PyTorch**.

Architecture:

```text
Input Layer
    ↓
Fully Connected Layer (16 neurons)
    ↓
ReLU
    ↓
Fully Connected Layer (8 neurons)
    ↓
ReLU
    ↓
Output Layer (1 neuron)
    ↓
Sigmoid
```

Training configuration:

- Optimizer: Adam
- Learning rate: `0.01`
- Loss function: Binary Cross Entropy (`BCELoss`)
- Batch size: `32`
- Epochs: `40`

The model outputs a probability between 0 and 1. A threshold of `0.5` is used to classify the customer:

```text
Probability >= 0.5 → Churn
Probability < 0.5  → No Churn
```

---

## Model Evaluation

The models are evaluated using:

### Accuracy

Measures the proportion of correct predictions.

### Precision

Measures how many customers predicted as churners are actually churners.

### Recall

Measures how many actual churners are successfully detected.

### F1-Score

The harmonic mean of precision and recall.

For a churn prediction problem, **Recall and F1-Score can be especially important**, because failing to identify a customer who is actually going to churn may represent a significant business opportunity loss.

The notebook also generates a **confusion matrix** for every model.

---

## Interactive Gradio Dashboard

The project includes a Gradio interface where the user can enter:

- Model
- Age
- Tenure
- Monthly Charges
- Total Charges
- Gender
- Contract Type
- Technical Support
- Internet Service

The dashboard returns:

```text
Prediction: Churn / No Churn
Churn Probability: XX.XX%
```

Available models in the interface:

- Neural Network
- KNN
- Random Forest
- SVM

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Option 1 — Google Colab

Upload the notebook:

```text
netflix.ipynb
```

Then run the cells from top to bottom.

The notebook installs Gradio automatically:

```python
pip install -q gradio
```

The final cell launches the interactive dashboard.

### Option 2 — Local Jupyter Environment

Install the requirements:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook
```

Run all cells sequentially.

---

## Example Prediction

Example customer:

```python
new_client = {
    'Age': 45,
    'Tenure': 12,
    'MonthlyCharges': 89.5,
    'TotalCharges': 1074.0,
    'Gender': 'Female',
    'ContractType': 'Month-to-month',
    'TechSupport': 'No',
    'InternetService': 'Fiber optic'
}
```

The system returns the selected model, predicted class, and estimated churn probability.

---

## Repository Structure

```text
customer-churn-prediction/
│
├── netflix.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations and synthetic data generation |
| Pandas | Data manipulation and preprocessing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning and evaluation |
| PyTorch | Neural Network / Deep Learning |
| Gradio | Interactive web interface |
| Google Colab | Development and experimentation environment |

---

## Reproducibility

Random seeds are fixed to improve reproducibility:

```python
np.random.seed(42)
torch.manual_seed(42)
```

The train/test split and Random Forest model also use `random_state=42`.

---

## Limitations

This version has several limitations that should be addressed before using it as a production system:

1. The dataset is synthetic.
2. The churn target is randomly generated rather than derived from real customer behavior.
3. Because the target is randomly generated, strong predictive relationships should not be interpreted as real business insights.
4. `LabelEncoder` is used for categorical features; production pipelines would typically use a more robust preprocessing pipeline such as `OneHotEncoder` with `ColumnTransformer`.
5. The trained models are not saved as model artifacts.
6. There is no hyperparameter optimization.
7. There is no cross-validation.
8. The current notebook does not include model explainability such as SHAP.
9. The Gradio application depends on the notebook's in-memory trained models and preprocessing objects.

---

## Future Improvements

Possible improvements include:

- Use a real customer churn dataset.
- Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Add cross-validation.
- Use a proper `Pipeline` and `ColumnTransformer`.
- Handle class imbalance with appropriate techniques.
- Add ROC-AUC and PR-AUC.
- Add feature importance analysis.
- Add SHAP explainability.
- Save trained models with `joblib` or PyTorch checkpoints.
- Separate training and inference code.
- Build a standalone web application.
- Deploy the application using Hugging Face Spaces, Render, or another hosting platform.
- Add automated tests.
- Add CI/CD.
- Add model monitoring for production use.

---

## Business Value

A production version of this system could help a subscription-based company:

- Identify customers at high risk of churn.
- Prioritize retention campaigns.
- Improve customer support targeting.
- Estimate churn probability for individual customers.
- Compare different predictive models before deployment.
- Reduce revenue loss caused by preventable customer churn.

---

## Disclaimer

This project is an educational Machine Learning / Deep Learning demonstration.

The current notebook uses synthetically generated data, so its predictions should **not** be interpreted as real-world customer churn predictions.

For real business deployment, the model should be retrained and validated on representative historical customer data.

---

## Author

**Ahmed Kamal**

GitHub: `https://github.com/YOUR_USERNAME`

LinkedIn: `https://www.linkedin.com/in/tasneem-said/`

---

## License

This project is intended for educational and portfolio purposes.
