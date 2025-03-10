# 🚢Titanic Survival Prediction

## 📌 Project Overview

This project analyzes the Titanic dataset to predict passenger survival using various machine learning models. The dataset is preprocessed, explored, and evaluated through multiple classification algorithms.

## 📂 Dataset

The titanic dataset used is this project is fetched from Seaborn:

- `titanic.csv`&#x20;
- Features include `age`, `fare`, `pclass`, `sex`, `embarked`, and others.

## ⚙️ Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly
- **ML Models:** Logistic Regression, KNN, Random Forest, SVM, Decision Tree, Naive Bayes
- **Other Tools:** Streamlit for visualization, GridSearchCV for hyperparameter tuning

## 🔍 Data Preprocessing

- Handling missing values (mean/mode imputation)
- Dropping columns having high missing values
- Encoding categorical variables using OneHotEncoding
- Normalization (MinMaxScaler, StandardScaler)

## 📊 Exploratory Data Analysis (EDA)

- Count plots for survival distribution
- Scatter plots for fare vs. age
- Histograms of fares for different passenger classes
- Boxplots for age and fare distribution
- FacetGrid visualizations for survival analysis

## 🚀 Model Training & Evaluation

Models were trained using a pipeline approach:

1. **Preprocessing** (Scaling + Encoding)
2. **Splitting Data** (80% train, 20% test)
3. **Model Training** (Logistic Regression, KNN, etc.)
4. **Hyperparameter Tuning** (GridSearchCV for Logistic Regression)
5. **Evaluation Metrics:**
   - Accuracy
   - Precision, Recall, F1 Score
   - Confusion Matrix
   - Cross-validation scores

## 🔥 Best Model Selection

- The best model was **Logistic Regression**, achieving the highest accuracy of 83.7079%.
- Logistic Regression with hyperparameter tuning performed well with an accuracy of 84.26966%.

## 📌 Deployment

- The trained model is saved using `joblib` (`titanic_trained_model.pkl`).
- The trained model is tested using sample data.
- Streamlit is used for interactive visualizations.
- Confusion matrices, survival rate visualizations, and EDA graphs are included.

## 📂 File Structure

```
|Titanic-Insights/
   |dashboard/
      |-- dataset/
         |-- titanic.csv
         |-- cleaned_data.csv
      |-- notebook/
         |-- titanic.ipynb
         |-- titanic_trained_model.pkl
   |-- app.py  # Streamlit app
   |-- requirements.txt
|-- README.md
```

## 🔧 Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/UFAQUE123/Titanic-Insights.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## ✨ Conclusion

- Feature engineering and proper preprocessing significantly improve model performance.
- Logistic Regression is the best-performing model for this dataset.
- The project demonstrates an end-to-end machine learning pipeline, from data preprocessing to deployment.

🚀 **Future Work:** 
Feature selection for better interpretability, and deploying a web-based ML model interface.

---

📌 **Author:** UFAQUE SHADAB\
📧 Contact: ufaqueshad127@gmail.com


