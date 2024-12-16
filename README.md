# Customer Churn Prediction Project

## 📌 Project Overview
This end-to-end project predicts customer churn using machine learning techniques. The workflow includes exploratory data analysis (EDA), feature importance analysis, optimized Random Forest modeling, and deployment of a Streamlit app for real-time predictions.

---

## 🚀 Features
- **Exploratory Data Analysis (EDA):** Visualizations to understand data trends and feature relationships.
- **Feature Engineering:** Handling missing data, scaling numerical features, and encoding categorical data.
- **Model Building:** Training and optimizing Logistic Regression, Decision Tree, and Random Forest models.
- **Hyperparameter Tuning:** GridSearchCV used to optimize Random Forest performance.
- **Model Deployment:** Deploy a Streamlit app to predict churn interactively.
- **Feature Importance:** Identify key factors influencing customer churn.

---

## 📊 Results
- **Optimized Model:** Random Forest Classifier
- **Performance Metrics:**
  - **Accuracy:** 99.95%
  - **Precision:** 99.93%
  - **Recall:** 99.98%
  - **F1-Score:** 99.95%
  - **ROC-AUC:** 99.999%

The model delivers high precision and recall, making it effective for identifying churn-prone customers.

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - Data Analysis: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-Learn, Imbalanced-Learn
  - Deployment: Streamlit
- **Tools:** GitHub, Jupyter Notebook

---

## 📂 Project Structure
```plaintext
customer_churn_prediction/
├── data/                        # Raw and processed datasets
│   ├── raw/                     # Raw data
│   └── processed/               # Cleaned and processed data
├── notebooks/                   # Jupyter Notebooks for EDA and modeling
│   ├── eda.ipynb                # EDA and feature exploration
│   ├── preprocessing.ipynb      # Data preprocessing and feature engineering
│   └── model_training.ipynb     # Model training and evaluation
├── scripts/                     # Python scripts for automation
│   ├── data_preprocessing.py    # Preprocessing pipeline
│   ├── model_training.py        # Model training and tuning
│   └── app.py                   # Streamlit app for deployment
├── models/                      # Saved machine learning models
│   └── optimized_rf_model.pkl   # Optimized Random Forest model
├── results/                     # Results and visualizations
│   ├── feature_importance.png   # Feature importance chart
│   └── metrics_summary.csv      # Model performance metrics
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

---

## ⚙️ How to Run

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/customer_churn_prediction.git
cd customer_churn_prediction
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
```bash
streamlit run scripts/app.py
```

The app allows users to input customer details and predict churn in real-time.

---

## 📈 Key Visualizations
- Churn distribution.
- Correlation heatmaps.
- Feature importance analysis.

Example Feature Importance Chart:
![Feature Importance](results/feature_importance.png)

---

## 📋 License
This project is licensed under the MIT License.

---

## 🤝 Acknowledgments
- Special thanks to open-source libraries and the ML community for providing tools and inspiration.

---

## 💡 Future Improvements
- Deploy the app on Hugging Face Spaces for broader access.
- Integrate additional models for ensemble learning.
- Automate the pipeline for continuous updates.

---

Feel free to reach out for questions or collaborations! 🚀
