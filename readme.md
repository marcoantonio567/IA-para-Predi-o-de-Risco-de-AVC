# 🧠 Stroke Prediction – AI for Stroke Risk Prediction

This project uses **Machine Learning** to predict the **probability of a person suffering a stroke** based on clinical, demographic, and behavioral characteristics.

The dataset used is the **Stroke Prediction Dataset** available on Kaggle.

---

## 🎯 Project Objective

The main objective of this project is to **train an AI model capable of estimating the probability (%) of an individual having a stroke**, using structured data that includes classic risk factors such as age, hypertension, cardiac history, blood glucose, among others.

The goal is:

* To identify the **most relevant factors** for stroke risk
* To create an **efficient and interpretable predictive model**
* To use metrics such as **AUC-ROC, Recall, Precision, and F1-score**
* To allow healthcare professionals or triage systems to perform **automated risk assessment**

---

## 📊 Dataset Column Descriptions

Below is a complete breakdown of each column in the dataset.

### **1. id**

* **Description:** Unique identifier for each patient.

* **Use:** Reference only.

* **Importance to the model:** Generally discarded, as it has no predictive value.

---

### **2. gender**

* **Description:** Biological sex of the patient.

* **Possible values:** `"Male"`, `"Female"`, `"Other"`.

* **Relevance:** May influence stroke risk due to physiological and epidemiological factors.

---

### **3. age**

* **Description:** Patient's age (numerical value).

* **Relevance:** It is one of the most important factors — stroke risk increases dramatically with age.

---

### **4. hypertension**

* **Description:** Indicates whether the patient has hypertension.

* **Values:**

* **0** = not hypertensive
* **1** = hypertensive
* **Relevance:** Hypertension is one of the biggest risk factors for stroke.

---

### **5. heart_disease**

* **Description:** Indicates the presence of heart disease.

* **Values:**

* **0** = no heart disease

* **1** = has heart disease
* **Relevance:** Highly relevant, as cardiovascular diseases are directly associated with stroke risk. ---

### **6. ever_married**

* **Description:** Indicates whether the person has ever been married.

* **Values:** `"Yes"` or `"No"`
* **Relevance:** Low. Generally not directly related to stroke and can be discarded in the model.

--

### **7. work_type**

* **Description:** Patient's occupation type.

* **Possible values:**

* `"Private"`

* `"Self-employed"`

* `"Govt_job"`

* `"Children"`

* `"Never_worked"`

* **Relevance:** May reflect lifestyle and routine — moderately relevant.

--

### **8. Residence_type**

* **Description:** Place of residence.

* **Values:** `"Urban"` or `"Rural"`
* **Relevance:** May indicate access to health services and environmental risk profile.

---

### **9. avg_glucose_level**

* **Description:** Average blood glucose level.

* **Relevance:** Highly relevant — elevated values ​​indicate a risk of diabetes, which increases the chances of stroke.

---

### **10. bmi**

* **Description:** Body Mass Index.

* **Relevance:** Represents obesity, sedentary lifestyle, and metabolic status — factors relevant to stroke.

---

### **11. smoking_status**

* **Description:** Patient's smoking status.

* **Values:**

* `"formerly smoked"`

* `"never smoked"`

* `"smokes"`

* `"Unknown"`
* **Relevance:** Extremely important. Smoking is a strong risk factor.

---

### **12. Stroke**

* **Description:** Indicates whether the patient has suffered a stroke.

* **Values:**

* **0** = has not had a stroke

* **1** = has had a stroke

* **Use:** **It is the target variable** of the AI ​​model.

---

## 🧪 Project Pipeline

1. **Data Cleaning**

* Remove/adjust missing values

* Handle categories

* Remove outliers (e.g., BMI)

* Analyze correlations

2. **Model Training**

* Test with **3 regression algorithms**

* **Winning model:** Random Forest (best performance in metrics)

3. **Validation**

* RMSE (Root Mean Squared Error)

* MAE (Mean Absolute Error)

* R² (Coefficient of determination)

* Feature Importance

4. **Prediction**

* Given a patient → model returns probability (%) of stroke.
---
### 🧠 Evaluated Models

- `RandomForestRegressor` — a set of decision trees trained with random samples. Captures non-linear relationships and interactions between variables, is robust to outliers, and works well as a general model. It was the winner in regression metrics.

- GradientBoostingRegressor — a sequence of trees that corrects errors from the previous model (boosting). Excellent for capturing complex patterns with high precision, but more sensitive to hyperparameters and the risk of overfitting.

- LinearRegression (baseline) — a linear model that assumes a linear relationship between variables. Simple and interpretable, useful as a reference; may not capture non-linear relationships well.

- Selection criterion: c
