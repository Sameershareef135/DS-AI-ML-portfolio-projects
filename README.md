# 🧠 DS-AI-ML Portfolio Projects

This repository showcases my **Data Science** and **Machine Learning** projects.  
Each folder contains an independent project with code, models, and deployment-ready Streamlit apps.

---

## 📊 Projects

### 1. 🩺 Diabetes Prediction System

**Description:**  
A machine learning model that predicts diabetes risk using patient health metrics.  
Built with Gradient Boosting Classifier trained on the Pima Indians Diabetes dataset with a 71% recall rate.

**Live App:**  
🔗 [https://ds-ai-ml-portfolio-projects-slwsfgqtey6wrutesj488v.streamlit.app/](https://ds-ai-ml-portfolio-projects-slwsfgqtey6wrutesj488v.streamlit.app/)

**Technologies:**  
Python, Streamlit, Scikit-learn, Pandas, Matplotlib, Seaborn

**Features:**  
- Interactive prediction form  
- Feature importance visualization  
- Model performance metrics  
- Real-time predictions  

**Model Performance:**  
- Accuracy: 73%  
- Precision: 61%  
- Recall: 71%

---

### 2. 🧑‍💼 HR Attrition Predictor

**Description:**  
Predicts whether an employee will leave the company based on HR analytics data.  
Trained using Logistic Regression and deployed as an interactive Streamlit web app.

**Live App:**  
🔗 *(Add Streamlit link after deployment)*

**Model Metrics:**  
- Accuracy: 76%  
- Recall (Leavers): 76%  
- AUC: 0.83  

**Technologies:**  
Python, Streamlit, Scikit-learn, Pandas, Matplotlib, Seaborn

---
### 📈 Stock Market Clustering (S&P 500)

**Description:**
An unsupervised machine learning project that clusters S&P 500 stocks based on historical market behavior.
Using 5 years of price data from Kaggle, weekly log returns were extracted and standardized before dimensionality reduction using Principal Component Analysis (PCA).
A Bayesian Gaussian Mixture Model (BGMM) was trained to identify probabilistic stock clusters, offering interpretable insights into sector-level relationships.

**Visualizations:**

PCA scatter plot of clustered stocks

Posterior probability heatmap (soft cluster memberships)

Confusion matrix comparing clusters with actual industry sectors

**Model Metrics:**

Effective Clusters: 6

PCA Variance Explained (PC1–3): 26.9%

Adjusted Rand Index (Sector Alignment): 0.04

Silhouette Score: 0.18

**Technologies:**
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

## ⚙️ How to Run Locally

1. Clone the repo  
   ```bash
   git clone https://github.com/Sameershareef135/DS-AI-ML-portfolio-projects.git
