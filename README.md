# Waze-Customer-Churn-Prediction

> **Google Advanced Data Analytics Professional Certificate — Capstone Project**

---

## Project Overview

Waze is a free, community-powered GPS navigation app used by millions of drivers every day. Like any product that depends on an active user base, one of Waze's core business concerns is **churn**: it's when a user stops using the app altogether.

The goal of this project was to build a machine learning model that could predict *which users are likely to churn*, so that Waze's team could proactively reach out and retain them before they leave. This is a classic binary classification problem: given a user's activity data (sessions, drives, kilometers traveled, etc.), can we predict whether they'll churn or be retained?

From a business perspective, identifying at-risk users early is a lot more cost-effective than trying to win them back after they've already gone. Getting this right could directly inform Waze's retention strategy — things like targeted in-app messaging or personalized nudges.

---

## Tools & Methods

**Language & Libraries**

| Purpose | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical testing | `scipy.stats` |
| Machine learning | `scikit-learn` |
| Gradient boosting | `xgboost` |
| Model persistence | `pickle` |

**Modeling Approach**

I built and compared three models:

1. **Logistic Regression**: my baseline. Simple, interpretable, and a good sanity check before moving to more complex approaches. It's essentially asking: "can we draw a straight line between churned and retained users?"

2. **Random Forest**: an ensemble model that builds hundreds of decision trees and combines their votes. I chose this because it handles non-linear relationships well and is generally more robust than a single decision tree. It also gives us feature importance scores, which are great for explaining results to a non-technical audience.

3. **XGBoost**: a more sophisticated gradient boosting algorithm that iteratively corrects its own errors. It tends to perform really well on tabular data like this.

For the two ensemble models, I used `GridSearchCV` with 4-fold cross-validation to tune hyperparameters. things like tree depth, the number of estimators, and minimum samples per leaf. I set recall as the primary optimization metric, because in a churn context it's worse to *miss* a churner than to occasionally flag someone who wasn't going to churn anyway.

**Feature Engineering**

I added several new features to give the models more signal:

- `km_per_driving_day`: how far a user drives on days they actually use the app
- `km_per_drive` and `km_per_hour`: measures of drive intensity
- `percent_sessions_in_last_month`: whether recent activity is high relative to historical use
- `total_sessions_per_day`: overall engagement rate since onboarding
- `percent_of_drives_to_favorite`: how routine vs. exploratory a user's driving is
- `professional_driver`: a binary flag for users with 60+ drives and 15+ driving days (possible couriers/delivery drivers)

---

## Key Insights & Results

**Class Imbalance**

The dataset is moderately imbalanced, where the majority of users are retained, with churned users making up a smaller portion. This was something I had to be mindful of throughout (more on that in the next section).

**What the EDA Revealed**

Comparing median feature values between churned and retained users surfaced some interesting patterns:

- **Churned users tend to drive more intensively**: higher kilometers per drive and more sessions in the last month relative to their historical total. This suggests some users may be "burning out" or using the app heavily for a short period before dropping off.
- **The `professional_driver` segment behaved differently** from casual users, which is why I created that feature.
- **Device type (iPhone vs. Android) didn't show a statistically significant difference** in mean drives, based on a Welch's t-test (p > 0.05). So device wasn't a meaningful churn predictor.

**Model Performance**

All three models were evaluated on a held-out validation set, with the champion then evaluated on a final test set. Because I optimized for recall, the tree-based models (Random Forest and XGBoost) generally outperformed Logistic Regression at identifying churned users, though sometimes at the cost of precision. The champion model was selected as whichever had the highest recall on the validation set, and its final metrics (accuracy, precision, recall, F1, and AUC-ROC) were reported on the test set.

The feature importance plots from the champion model pointed to engagement-rate features — particularly `km_per_driving_day`, `percent_sessions_in_last_month`, and `total_sessions_per_day` — as the strongest predictors of churn.

---

## What I Learned

This project was part of my journey through the **Google Advanced Data Analytics Professional Certificate**. Here are my biggest takeaways:

**Choosing the right metric**: My first instinct was to optimize for accuracy. But with an imbalanced dataset, a model can hit decent accuracy just by predicting "retained" for almost everyone, and completely miss actual churners. Switching to recall as my primary metric during hyperparameter tuning changed everything. It forced the model to actually try to find churned users, not just the easy majority.

**Hyperparameter tuning is time-consuming but worth it.** Running `GridSearchCV` across two models with multiple parameters took a while, and I learned pretty quickly that you have to be selective about your search space. Throwing in too many options makes the grid explode combinatorially. I ended up narrowing down which parameters matter most for each model (e.g., `max_depth` and `n_estimators` for Random Forest; `learning_rate` and `min_child_weight` for XGBoost) rather than searching everything.

**Feature engineering can do a lot of the heavy lifting.** The raw dataset columns were useful, but the engineered features — especially the ratio-based ones like `km_per_driving_day` — ended up being among the most predictive. It reminded me that domain thinking (asking *"what would actually signal churn from a user behavior standpoint?"*) is just as important as algorithm selection.

**Handling infinity values is annoying and necessary.** When I created `km_per_driving_day` by dividing by `driving_days`, some users had zero driving days, which produced `inf`. I had to explicitly replace those with 0. Small thing, but the kind of real-world messiness that doesn't come up in toy datasets.

**Stratified splitting is non-negotiable with imbalanced classes.** I made sure to use `stratify=y` in both the train/test and train/validation splits so that the class ratio was preserved across all sets. Without that, you risk training or evaluating on a split that doesn't represent the actual distribution — which would make your metrics misleading.

---

*This project is part of my Google Advanced Data Analytics Professional Certificate portfolio. The dataset used is the Waze case study dataset provided through the certificate program.*
