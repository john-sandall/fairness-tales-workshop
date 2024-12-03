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
from sklearn import (
    dummy,
    ensemble,
    model_selection,
    tree,
)
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(".")
np.set_printoptions(legacy="1.25")


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
        .replace("—", " ")
    )


# %%
df = pd.read_feather(ROOT / "data" / "output" / "resumes.feather").reset_index(
    drop=True
)

df["text"] = df["cv"].apply(clean_cv_text)

df.shape

# %% [markdown]
# ## Let's build a (very) biased model!

# %%
X = df[["text"]]
y = df["callback"]

# %%
vectorizer = CountVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english")

vectorizer.fit(X.text)
X_text = pd.DataFrame(
    vectorizer.transform(X.text).toarray(), columns=vectorizer.get_feature_names_out()
)
X = pd.concat([X.drop(columns="text"), X_text], axis=1)
X.head()

# %%
f"Baseline callback rate is {y.mean():.1%}"

# %% [markdown]
# ## Let's fit a small decision tree for interpretability

# %%
model = tree.DecisionTreeClassifier(max_depth=4)
model = model.fit(X, y)

plt.figure(figsize=(15, 8))
tree.plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=9);

# %%
sns.barplot(y=y, x=df.sex)

# %%
sns.barplot(x=(X.aws > 0), y=y, hue=df.sex)

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
aequitas = (
    y.reset_index().rename(columns={"callback": "label_value"}).drop(columns="index")
)
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
from fairlearn.reductions import ErrorRate, EqualizedOdds, ExponentiatedGradient

# %% [markdown]
# ## Prepare train, test, and protected characteristics dataframes

# %%
A_str = df.sex.astype("category")
A_str.value_counts(normalize=True)

# %%
y.value_counts(normalize=True)

# %%
X_train, X_test, y_train, y_test, A_train, A_test = model_selection.train_test_split(
    X, y, A_str, test_size=0.35, stratify=y
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
mf_unmitigated.by_group.plot.bar(
    subplots=True, layout=[1, 3], figsize=[12, 4], legend=None, rot=0
);

# %%
mf_unmitigated.difference()

# %%
balanced_accuracy_unmitigated = balanced_accuracy_score(y_test, y_pred)
balanced_accuracy_unmitigated

# %%
equalized_odds_unmitigated = equalized_odds_difference(
    y_test, y_pred, sensitive_features=A_test
)
equalized_odds_unmitigated

# %% [markdown]
# ## Mitigate using post-processing techniques (ThresholdOptimizer)

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
mf_postprocess.by_group.plot.bar(
    subplots=True, layout=[1, 3], figsize=[12, 4], legend=None, rot=0
)

# %%
mf_postprocess.difference()

# %%
balanced_accuracy_postprocess = balanced_accuracy_score(y_test, y_pred_postprocess)
balanced_accuracy_postprocess

# %%
equalized_odds_postprocess = equalized_odds_difference(
    y_test, y_pred_postprocess, sensitive_features=A_test
)
equalized_odds_postprocess

# %% [markdown]
# ## Mitigate using Fairlearn Reductions

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
mf_reduction.by_group.plot.bar(
    subplots=True, layout=[1, 3], figsize=[12, 4], legend=None, rot=0
)

# %%
mf_reduction.difference()

# %%
balanced_accuracy_reduction = balanced_accuracy_score(y_test, y_pred_reduction)
balanced_accuracy_reduction

# %%
equalized_odds_reduction = equalized_odds_difference(
    y_test, y_pred_reduction, sensitive_features=A_test
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
"""
)

# %%
print(
    f"""
{equalized_odds_unmitigated=}
{equalized_odds_postprocess=}
{equalized_odds_reduction=}
"""
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
    }
).plot(x="error", y="equalized_odds", title="The Performance-Fairness Trade-Off");

# %%
