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

## Summary of Findings

This project unfolded across multiple milestones in the certificate program, each building on the last. Here's a consolidated look at what the most important stages revealed and how the findings evolved.

---

### Milestone 4 — Two-Sample Hypothesis Test

The first major analytical question was: **does device type actually affect how much users drive?**

iPhone users averaged 68 drives compared to 66 for Android users — a small raw difference. But after running a **Welch's two-sample t-test** (which doesn't assume equal variance between groups, a safer choice here), the result was clear: **the difference is not statistically significant** at the α = 0.05 level.

What does that mean in practice? Device type alone can't explain churn behavior. A user's platform isn't predictive — so any retention strategy that tried to target iPhone or Android users differently, based on this variable alone, wouldn't be grounded in data.

The recommendation from this milestone was to run additional t-tests on other behavioral variables to build a fuller picture. It was a good reminder that a non-significant result is still a finding — it rules things out and points us toward where to look next.

---

### Milestone 5 — Binomial Logistic Regression

With hypothesis testing done, the next step was building an actual predictive model. I started with **binomial logistic regression** — a natural first choice for binary classification (churned vs. retained) because it's interpretable and establishes a performance baseline.

The results were honest about the model's limitations:

- **Precision: 53%** — just over half of predicted churners actually churned
- **Recall: 9%** — the model only identified about 1 in 11 actual churners

That recall score is rough. The confusion matrix showed a large number of false negatives — users who churned but were classified as retained. For a business trying to proactively intervene, missing 91% of churners is a serious problem.

Still, the model surfaced something genuinely useful: **`activity_days` was by far the most important feature**, with a *negative* correlation with churn (more active days = lower churn risk). This makes intuitive sense and gave us a strong signal to carry into more complex models. Interestingly, `km_per_driving_day` showed a *positive* correlation with churn — users who drive farther on the days they do use the app are actually *more* likely to churn, which is a counterintuitive finding worth investigating further.

The conclusion from this milestone: logistic regression isn't powerful enough to rely on here, but it validated the feature engineering approach and confirmed that **more data — especially drive-level granular data like timestamps, geographic locations, and route types — would meaningfully improve any future model**.

---

### Milestone 6 — Random Forest & XGBoost (Champion Model)

This was the main event. I built two tree-based ensemble models and compared them against each other (and implicitly against the logistic regression baseline).

**Key results:**

- **XGBoost outperformed Random Forest** across all evaluation metrics and fit the data better overall
- The XGBoost model's **recall score of ~17%** was nearly double that of the logistic regression model — still modest in absolute terms, but a meaningful improvement given the same dataset
- Accuracy and precision remained comparable to the logistic regression model, so the gain in recall didn't come at a dramatic cost elsewhere

**What the feature importance chart revealed:**

The top predictors of churn — by a significant margin — were the engineered features, not the raw dataset columns:

| Rank | Feature | Type |
|---|---|---|
| 1 | `km_per_hour` | Engineered |
| 2 | `n_days_after_onboarding` | Raw |
| 3 | `percent_sessions_in_last_month` | Engineered |
| 4 | `total_sessions_per_day` | Engineered |
| 5 | `duration_minutes_drives` | Raw |
| 6 | `percent_of_drives_to_favorite` | Engineered |

**The bigger picture from Milestone 6:**

Even with the best models I could build, the results confirm what the logistic regression already hinted at: **the current dataset is not sufficient to reliably predict churn**. The models demonstrate that the *right kind* of data matters — specifically, the team would benefit from drive-level data (individual trip timestamps, geographic coordinates, route types), more granular session data, and ideally a longer observation window per user.

The recommendation from this milestone was a second iteration of the project with richer data collection in place. The ensemble models are more valuable than a single logistic regression, but they're not a production-ready solution yet — they're a proof of concept that shows what's possible with better inputs.

---

### Overall Project Takeaway

Across all three milestones, the consistent theme was this: **the data we have tells a partial story, but not enough of one to act on confidently**. The hypothesis test ruled out device type. The logistic regression flagged activity days and drive distance as meaningful signals. The ensemble models confirmed those signals and squeezed more predictive power out of them — but the ceiling is limited by what the dataset actually captures.

If this were a real project, the next step wouldn't be a fancier algorithm — it would be better data collection. And honestly, learning to say that clearly and back it up with evidence feels like one of the most important skills this certificate has helped me build.

## What I Learned

This project was part of my journey through the **Google Advanced Data Analytics Professional Certificate**. Here are my biggest takeaways:

**Choosing the right metric**: My first instinct was to optimize for accuracy. But with an imbalanced dataset, a model can hit decent accuracy just by predicting "retained" for almost everyone, and completely miss actual churners. Switching to recall as my primary metric during hyperparameter tuning changed everything. It forced the model to actually try to find churned users, not just the easy majority.

**Hyperparameter tuning is time-consuming but worth it.** Running `GridSearchCV` across two models with multiple parameters took a while, and I learned pretty quickly that you have to be selective about your search space. Throwing in too many options makes the grid explode combinatorially. I ended up narrowing down which parameters matter most for each model (e.g., `max_depth` and `n_estimators` for Random Forest; `learning_rate` and `min_child_weight` for XGBoost) rather than searching everything.

**Feature engineering can do a lot of the heavy lifting.** The raw dataset columns were useful, but the engineered features — especially the ratio-based ones like `km_per_driving_day` — ended up being among the most predictive. It reminded me that domain thinking (asking *"what would actually signal churn from a user behavior standpoint?"*) is just as important as algorithm selection.

**Handling infinity values is annoying and necessary.** When I created `km_per_driving_day` by dividing by `driving_days`, some users had zero driving days, which produced `inf`. I had to explicitly replace those with 0. Small thing, but the kind of real-world messiness that doesn't come up in toy datasets.

**Stratified splitting is non-negotiable with imbalanced classes.** I made sure to use `stratify=y` in both the train/test and train/validation splits so that the class ratio was preserved across all sets. Without that, you risk training or evaluating on a split that doesn't represent the actual distribution — which would make your metrics misleading.

---

*This project is part of my Google Advanced Data Analytics Professional Certificate portfolio. The dataset used is the Waze case study dataset provided through the certificate program.*
