# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (fairness-tales-workshop)
#     language: python
#     name: fairness-tales-workshop
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# <img src="https://static1.squarespace.com/static/5ba26f9d89c1720405dcfae2/t/5bbc69570d929721d5a5ff2c/1726236705071/" width=300>
#
# <h1>PyData Global 2024</h1>
# <h2>Fairness Tales: How To Measure And Mitigate Unfair Bias in Machine Learning Models</h2>
# <h3>Notebook 2 - Modelling</h3>
#
# This notebook demonstrates how bias can emerge in machine learning models and explores different techniques to mitigate it. Using a resume screening dataset, we:
#
# 1. Train an intentionally biased model to show how ML systems can perpetuate discrimination
# 2. Evaluate the model's fairness metrics across different demographic groups
# 3. Apply various fairness-aware techniques (including Fairlearn) to mitigate the discovered biases
# 4. Compare the performance-fairness tradeoffs between different approaches
#
# The goal is to illustrate both the potential pitfalls in ML systems and practical approaches to building more equitable models.
#
# ---

# %% [markdown]
# In this workshop, we'll deliberately create and then fix a biased model. This helps us:
# 1. Understand how bias can emerge in ML systems
# 2. Learn to identify bias through metrics
# 3. Practice different approaches to mitigate unfairness
#
# Remember: In a real application, we would never intentionally create a biased model. This is purely for educational purposes.

# %%
# %pwd

# %%
# %cd ~/code/fairness-tales-workshop/

# %%
# %load_ext jupyter_black
# %load_ext autoreload
# %autoreload 2

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Section 1: Let's train a (very) biased model!

# %% [markdown]
# ## Imports

# %%
import string
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# sklearn
from sklearn import dummy, ensemble, model_selection, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

ROOT = Path()
np.set_printoptions(legacy="1.25")

# %% [markdown]
# We'll use a variety of libraries for this analysis:
# - `lightgbm` for our main classifier
# - Standard data science tools (`numpy`, `pandas`, `sklearn`)
# - `fairlearn` for bias mitigation (we'll use this later)

# %% [markdown]
# ## Load data


# %%
def whitespace_cleaner(s, n=8):
    """Worse I've seen is a CV full of whitespace, n=8 was enough for this."""
    for i in range(n, 0, -1):
        s = s.replace(" " * i, " ")
    return s


def clean_cv_text(text):
    return whitespace_cleaner(
        text.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .replace("•", " ")
        .replace("\n", " ")
        .replace("*", " ")
        .replace("-", " ")
        .replace("—", " "),
    )


# %%
df = pd.read_feather(ROOT / "data" / "output" / "resumes.feather").reset_index(drop=True)

df["text"] = df["cv"].apply(clean_cv_text)

df.shape

# %% [markdown]
# We're working with resume data where we want to predict whether a candidate received a callback. The data includes the CV text and demographic information like gender.
#
# First, we'll clean the CV text by:
# - Converting to lowercase
# - Removing punctuation and special characters
# - Standardizing whitespace

# %% [markdown]
# ## Let's build a (very) biased model!

# %%
X = df[["text"]]
y = df["callback"]

# %%
vectorizer = CountVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english")

vectorizer.fit(X.text)
X_text = pd.DataFrame(
    vectorizer.transform(X.text).toarray(),
    columns=vectorizer.get_feature_names_out(),
)
X = pd.concat([X.drop(columns="text"), X_text], axis=1)
X.head()

# %% [markdown]
# To convert our text data into features, we use a CountVectorizer to:
# 1. Extract the top 200 most common words/phrases (unigrams and bigrams)
# 2. Remove common English stop words
# 3. Create a sparse matrix where each column represents a word/phrase frequency

# %%
f"Baseline callback rate is {y.mean():.1%}"

# %% [markdown]
# ## Let's fit a small decision tree for interpretability

# %%
model = tree.DecisionTreeClassifier(max_depth=4)
model = model.fit(X, y)

plt.figure(figsize=(15, 8))
tree.plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=9)
# %%
sns.barplot(y=y, x=df.sex)

# %%
sns.barplot(x=(X.aws > 0), y=y, hue=df.sex)

# %% [markdown]
# Let's start with a simple decision tree to visualize how the model makes decisions. We'll intentionally keep it shallow (max_depth=4) for interpretability.

# %% [markdown]
# ## Find our next top (biased) model

# %%
model_selection.cross_val_score(
    dummy.DummyClassifier(),
    X=X,
    y=y,
    cv=5,
    scoring="accuracy",
).mean()

# %%
model_selection.cross_val_score(
    ensemble.RandomForestClassifier(n_estimators=100),
    X=X,
    y=y,
    cv=5,
    scoring="accuracy",
).mean()

# %%
model_selection.cross_val_score(
    tree.DecisionTreeClassifier(),
    X=X,
    y=y,
    cv=5,
    scoring="accuracy",
).mean()

# %%
model_selection.cross_val_score(
    tree.DecisionTreeClassifier(max_depth=4),
    X=X,
    y=y,
    cv=5,
    scoring="accuracy",
).mean()

# %%
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 10,
    "max_depth": 3,
    "n_jobs": 1,
    "verbose": -1,
}

model = lgb.LGBMClassifier(**lgb_params)

model_selection.cross_val_score(
    model,
    X=X,
    y=y,
    cv=5,
    scoring="accuracy",
).mean()

# %%
model = lgb.LGBMClassifier(**lgb_params).fit(X, y)

# %% [markdown]
# ## Interpret the LGBM classifier using permutation importance

# %%
r = permutation_importance(model, X, y, n_repeats=30)

# %%
(
    pd.DataFrame({"feature": X.columns, "importance": r["importances_mean"]})
    .sort_values("importance")
    .set_index("feature")
    .tail(10)
    .plot(kind="barh")
)

# %%
sns.barplot(x=(X["architect"] > 0), y=y, hue=df.sex)

# %% [markdown]
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Section 2: Export for Aequitas

# %%
aequitas = y.reset_index().rename(columns={"callback": "label_value"}).drop(columns="index")
aequitas["label_value"] = y
aequitas["score"] = model.predict(X)
aequitas["sex"] = df.sex

aequitas.to_csv(ROOT / "data" / "output" / "aequitas.csv", index=False)

# %% [markdown]
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Section 3: Intro to Fairlearn

# %%
# fairlearn
from fairlearn.metrics import (
    MetricFrame,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ErrorRate, ExponentiatedGradient

# %% [markdown]
# ## Understanding Fairlearn
#
# Fairlearn is a Python package that helps assess and mitigate unfairness in machine learning models. It provides tools for:
# 1. Measuring model fairness through disaggregated metrics
# 2. Mitigating unfairness through various algorithmic interventions
# 3. Visualizing and comparing model performance across different demographic groups
#
# In this notebook, we'll explore two main mitigation approaches:
# - Post-processing with ThresholdOptimizer
# - In-processing with the Reductions approach
#
# Both techniques help us balance model performance with fairness constraints.

# %% [markdown]
# ## Understanding MetricFrame
#
# A MetricFrame is Fairlearn's core tool for disaggregated performance assessment. It allows us to:
# - Evaluate multiple metrics across different demographic groups
# - Compare model performance between groups
# - Quantify disparities in model behavior
#
# MetricFrame calculates:
# - Overall metrics (aggregate performance)
# - By-group metrics (performance for each demographic group)
# - Difference metrics (largest gap between any two groups)
#
# In our case, we're particularly interested in:
# - Balanced accuracy (model performance)
# - False positive/negative rates (error patterns)
# - Equalized odds difference (overall fairness measurement)

# %% [markdown]
# ## Prepare train, test, and protected characteristics dataframes

# %%
A_str = df.sex.astype("category")
A_str.value_counts(normalize=True)

# %%
y.value_counts(normalize=True)

# %%
X_train, X_test, y_train, y_test, A_train, A_test = model_selection.train_test_split(
    X,
    y,
    A_str,
    test_size=0.35,
    stratify=y,
)

# %%
X_train.shape

# %%
X_test.shape

# %% [markdown]
# ## Fit our LGBM model to the training data & generate predictions on test data

# %%
model = lgb.LGBMClassifier(**lgb_params)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %% [markdown]
# ## Evaluate the "raw" unmitigated LGBM model

# %%
fairness_metrics = {
    "balanced_accuracy": balanced_accuracy_score,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
}

# %%
mf_unmitigated = MetricFrame(
    metrics=fairness_metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test,
)

# %%
mf_unmitigated.overall

# %%
mf_unmitigated.by_group

# %%
mf_unmitigated.by_group.plot.bar(subplots=True, layout=[1, 3], figsize=[12, 4], legend=None, rot=0)
# %%
mf_unmitigated.difference()

# %%
balanced_accuracy_unmitigated = balanced_accuracy_score(y_test, y_pred)
balanced_accuracy_unmitigated

# %%
equalized_odds_unmitigated = equalized_odds_difference(y_test, y_pred, sensitive_features=A_test)
equalized_odds_unmitigated

# %% [markdown]
# ## Mitigate using post-processing techniques (ThresholdOptimizer)

# %% [markdown]
# ### Understanding ThresholdOptimizer
#
# ThresholdOptimizer is a post-processing technique that adjusts model predictions after training. Key points:
# - Takes an existing trained model and transforms its outputs
# - Finds different thresholds for each demographic group
# - Optimizes for a specified metric (like balanced accuracy) while satisfying fairness constraints
# - Requires access to sensitive features during both training and prediction
#
# In our case, we use it to optimize balanced accuracy while satisfying equalized odds constraints. This means we're trying to:
# 1. Maintain good overall prediction accuracy
# 2. Ensure similar false positive and false negative rates across gender groups
#
# Note: A key limitation is that we need the sensitive feature (gender) at prediction time.

# %%
postprocess_mitigator = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",  # Optimize FPR and FNR simultaneously
    objective="balanced_accuracy_score",
    prefit=True,
    predict_method="predict_proba",
)

# %%
postprocess_mitigator.fit(X=X_train, y=y_train, sensitive_features=A_train)
y_pred_postprocess = postprocess_mitigator.predict(X_test, sensitive_features=A_test)

# %%
mf_postprocess = MetricFrame(
    metrics=fairness_metrics,
    y_true=y_test,
    y_pred=y_pred_postprocess,
    sensitive_features=A_test,
)
mf_postprocess.overall

# %%
mf_postprocess.by_group

# %%
mf_postprocess.by_group.plot.bar(subplots=True, layout=[1, 3], figsize=[12, 4], legend=None, rot=0)

# %%
mf_postprocess.difference()

# %%
balanced_accuracy_postprocess = balanced_accuracy_score(y_test, y_pred_postprocess)
balanced_accuracy_postprocess

# %%
equalized_odds_postprocess = equalized_odds_difference(
    y_test,
    y_pred_postprocess,
    sensitive_features=A_test,
)
equalized_odds_postprocess

# %% [markdown]
# ## Mitigate using Fairlearn Reductions

# %% [markdown]
# ### Understanding the Reductions Approach
#
# The Reductions approach, implemented through ExponentiatedGradient, is an in-processing technique that enforces fairness during model training. Key aspects:
#
# 1. How it works:
#    - Creates a sequence of reweighted datasets
#    - Retrains the base classifier on each dataset
#    - Guaranteed to find a model satisfying fairness constraints while optimizing performance
#
# 2. Key differences from ThresholdOptimizer:
#    - Fairness is enforced during training, not after
#    - Doesn't need sensitive features at prediction time
#    - Results in a single model rather than group-specific thresholds
#
# 3. Parameters:
#    - epsilon: controls the maximum allowed disparity
#    - Recommended to set epsilon ≈ 1/√(number of samples)
#
# The algorithm produces multiple candidate models, allowing us to examine the performance-fairness trade-off and select the most appropriate model for our needs.

# %%
objective = ErrorRate(costs={"fp": 0.1, "fn": 0.9})
constraint = EqualizedOdds(difference_bound=0.01)
reduction_mitigator = ExponentiatedGradient(model, constraint, objective=objective)
reduction_mitigator.fit(X_train, y_train, sensitive_features=A_train)

# %%
y_pred_reduction = reduction_mitigator.predict(X_test)

# %%
mf_reduction = MetricFrame(
    metrics=fairness_metrics,
    y_true=y_test,
    y_pred=y_pred_reduction,
    sensitive_features=A_test,
)
mf_reduction.overall

# %%
mf_reduction.by_group

# %%
mf_reduction.by_group.plot.bar(subplots=True, layout=[1, 3], figsize=[12, 4], legend=None, rot=0)

# %%
mf_reduction.difference()

# %%
balanced_accuracy_reduction = balanced_accuracy_score(y_test, y_pred_reduction)
balanced_accuracy_reduction

# %%
equalized_odds_reduction = equalized_odds_difference(
    y_test,
    y_pred_reduction,
    sensitive_features=A_test,
)
equalized_odds_reduction

# %% [markdown]
# ## Compare our three models

# %%
mf_unmitigated.by_group

# %%
mf_postprocess.by_group

# %%
mf_reduction.by_group

# %%
print(
    f"""
{balanced_accuracy_unmitigated=}
{balanced_accuracy_postprocess=}
{balanced_accuracy_reduction=}
""",
)

# %%
print(
    f"""
{equalized_odds_unmitigated=}
{equalized_odds_postprocess=}
{equalized_odds_reduction=}
""",
)

# %%
pd.DataFrame(
    {
        "error": [
            -balanced_accuracy_unmitigated,
            -balanced_accuracy_postprocess,
            -balanced_accuracy_reduction,
        ],
        "equalized_odds": [
            equalized_odds_unmitigated,
            equalized_odds_postprocess,
            equalized_odds_reduction,
        ],
    },
).plot(x="error", y="equalized_odds", title="The Performance-Fairness Trade-Off")

# %% [markdown]
# We can see that LightGBM performs best in terms of accuracy. However, high accuracy doesn't mean the model is fair - it might be very accurate at perpetuating historical biases in the training data. This is why we need specialized fairness metrics and mitigation techniques.

# %% [markdown]
# ## Understanding the Trade-offs
#
# As we can see from the results:
# - The unmitigated model has the highest accuracy but shows significant bias
# - The post-processing approach (ThresholdOptimizer) reduces bias but with some cost to accuracy
# - The reductions approach provides a different balance between fairness and performance
#
# In practice, choosing between these approaches depends on:
# - Your specific fairness requirements
# - Whether you can access protected characteristics at prediction time
# - The acceptable trade-off between fairness and performance
# - Technical constraints of your deployment environment

# %% [markdown]
# ## Key Takeaways
#
# 1. Machine learning models can easily perpetuate or amplify existing biases
# 2. We need specialized tools and metrics to identify unfairness
# 3. Different mitigation techniques offer different trade-offs
# 4. There's usually no perfect solution - but we can significantly improve fairness
#
# Remember: Fairness in ML is an ongoing process, not a one-time fix. Regular monitoring and updates are essential to maintain fair outcomes.

# %%
