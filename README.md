# 🎧 Predict Podcast Listening Time

![Podcast Duck](https://img.freepik.com/fotos-premium/pato-de-desenho-animado-usando-fones-de-ouvido-e-ouvindo-musica_902846-155065.jpg?w=360)

Welcome to the **Predict Podcast Listening Time** project!  
This notebook explores data science techniques to predict how long a listener will engage with a podcast episode, based on metadata such as the episode's title, sentiment, host and guest popularity, number of ads, and more.

---

## 🧠 Project Overview

This project was developed for a Kaggle competition and is focused on **regression modeling**.  
The goal is to predict the listening time (`Listening_Time_minutes`) of podcast episodes using a mix of numerical, categorical, and text-based features.

The project is structured into the following key steps:

1. **Exploratory Data Analysis (EDA)**  
   Visualizations and profiling using `ydata-profiling` and `seaborn`.

2. **Missing Value Imputation**  
   Filling missing data using median imputation and checking for distribution anomalies.

3. **Outlier Treatment**  
   Filtering outliers based on IQR method to improve model stability.

4. **Target Variable Transformation (Yeo-Johnson)**  
   The `Listening_Time_minutes` is transformed using Yeo-Johnson, a power transformation method that can handle both positive and negative values and normalize skewed distributions.  
   This helps improve linear model performance.

5. **Normality Test - Shapiro-Francia**  
   A statistical test used to assess the normality of the target variable, particularly suitable for large datasets.  
   The normality assumption is important for linear models and justifies the need for transformation.

6. **Multicollinearity and Feature Relationship (Cramér's V)**  
   Cramér’s V is used to evaluate the association between two categorical variables — in this case, **Podcast_Name** and **Genre**.  
   High correlation prompted the removal of `Podcast_Name`.

7. **ANOVA and Tukey’s HSD Test**  
   These are used to test whether the **Episode_Sentiment** has a statistically significant effect on `Listening_Time_minutes`.  
   ANOVA checks for overall differences, while **Tukey’s HSD** performs post-hoc pairwise comparisons to find exactly which groups differ.

8. **Baseline Model (OLS Regression)**  
   Using statsmodels and a **stepwise selection** process to find an optimized linear regression model.

9. **Advanced Model (XGBoost)**  
   Tuned using `GridSearchCV` and evaluated with RMSE.  
   Predictions are back-transformed using the inverse of Yeo-Johnson before submission.

10. **Submission Preparation**  
    Predictions are formatted according to the competition's submission format.

---

## 📊 Main Tools & Libraries

- `pandas`, `numpy`
- `ydata-profiling`
- `scikit-learn`, `xgboost`
- `matplotlib`, `seaborn`
- `statsmodels`, `pingouin`
- `scipy`, `statstests` (custom for Shapiro-Francia)
- `PowerTransformer` (Yeo-Johnson)

---

## 📬 Contact

Feel free to open issues or pull requests!  
Project by **Claudio Sturaro** – [LinkedIn](https://www.linkedin.com/in/claudiosturaro)

---
