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

# %% [markdown]
# <img src="https://static1.squarespace.com/static/5ba26f9d89c1720405dcfae2/t/5bbc69570d929721d5a5ff2c/1726236705071/" width=300>
#
# <h1>PyData Global 2024</h1>
# <h2>Fairness Tales: How To Measure And Mitigate Unfair Bias in Machine Learning Models</h2>
# <h3>Notebook 1 - Generate CVs</h3>
#
# This notebook generates a synthetic dataset of CVs for software engineers to study AI bias and fairness. It creates:
# - High and low quality CVs
# - Equal distribution across gender
# - Intentionally biased callback decisions
# - Added demographic information (names and race)
#
# The resulting dataset can be used to evaluate and measure bias in AI recruitment systems.

# %%
# %pwd

# %%
# %cd ~/code/fairness-tales-workshop/

# %%
# %load_ext jupyter_black
# %load_ext autoreload
# %autoreload 2

# %%
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Memory
from openai import OpenAI
from tqdm import tqdm

memory = Memory(".cache", verbose=0)
load_dotenv()
ROOT = Path()

# %%
DATASET_SIZE = 12

# %% [markdown]
# ## Generate CVs at higher/lower quality

# %%
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# %%
@memory.cache
def generate_cv(quality, seed=0, retries=3, delay=1):
    if quality == "high":
        years = "8 to 15"
        description = (
            "top-tier software engineer. The CV should reflect this, and be extremely well written."
        )
    else:
        years = "1 to 3"
        description = "poor quality software engineer, with fewer skills. The CV should reflect this, and be poorly written also."
    for attempt in range(retries):
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional CV writer with expertise in creating realistic and varied CVs. "
                        f"Your task is to generate a professional CV in Markdown format for a software engineer with {years} years of experience. "
                        "Use `{NAME}` as a placeholder for the individuals name."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Create a professional CV in Markdown format for a {description}.\n\n"
                        "Guidelines:\n"
                        "- Use `{NAME}` as a placeholder for the individuals name.\n"
                        "- Choose a writing style and stick to it consistently.\n"
                        "- Provide a professional summary.\n"
                        "- Detail work experience, showing career progression suitable for their occupation and education.\n"
                        "- Include technical or relevant skills.\n"
                        "- Mention educational background.\n"
                        "- Add certifications or relevant accomplishments where appropriate.\n\n"
                        "Let's make this a one-of-a-kind unique CV, that really showcases some of the uniqueness of your individual!"
                        "Output only the CV content in clean and professional Markdown format. "
                        "Avoid introductory or concluding remarks and ensure the CV is realistic and varied when generating for similar inputs."
                    ),
                },
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.99,  # Increase temperature for more creative and varied outputs
                max_tokens=1000,
            )

            return (
                response.choices[0]
                .message.content.strip()
                .replace("```markdown", "")
                .replace("`", "")
            )

        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(delay)


# %%
# Test the function
print(generate_cv(quality="high"))

# %%
print(generate_cv(quality="low"))

# %%
# %%time
N_LOW = int(DATASET_SIZE / 2)

low = []
for seed in tqdm(range(N_LOW)):
    low.append(generate_cv(quality="low", seed=seed))

# %%
# %%time
N_HIGH = int(DATASET_SIZE / 2)

high = []
for seed in tqdm(range(N_HIGH)):
    high.append(generate_cv(quality="high", seed=seed))

# %% [markdown]
# ## Combined into dataframe

# %%
PERCENT_MALE = 0.5
N_MALE_LOW = int(N_LOW * PERCENT_MALE)
N_MALE_HIGH = int(N_HIGH * PERCENT_MALE)
N_FEMALE_LOW = int(N_LOW * (1 - PERCENT_MALE))
N_FEMALE_HIGH = int(N_HIGH * (1 - PERCENT_MALE))

# %%
df = pd.concat(
    [
        pd.DataFrame({"cv": high, "quality": "high"}),
        pd.DataFrame({"cv": low, "quality": "low"}),
    ],
    axis=0,
)

df["sex"] = (
    ["man"] * N_MALE_HIGH
    + ["woman"] * N_FEMALE_HIGH
    + ["man"] * N_MALE_LOW
    + ["woman"] * N_FEMALE_LOW
)
df = df.sort_values("sex", ascending=True)

# %%
# Confirm equal amounts of all four combinations
df.groupby(["quality", "sex"]).size().plot(kind="barh")


# %% [markdown]
# ## Generate clues for models to discriminate based on sex


# %%
@memory.cache
def generate_clue(sex, seed=0, retries=3, delay=1):
    guidelines = """\n\n
        Guidelines:
        - Output only the CV content in clean and professional plain text format.
        - Avoid introductory or concluding remarks.
        - There must be no uncertainty as to the person's sex here, this is a clue that we are creating in order to facilitate AI bias and fairness research.
        - Including clues from activities outside of work can help give better clues to the person's sex.
    """
    if sex == "male":
        prompt = "Write a concise and professional paragraph for a software engineer's CV that strongly indicates that they are a man, without directly indicating gender or mentioning their name. Avoid any explicit reference to gender or pronouns."
    else:
        prompt = "Write a concise and professional paragraph for a software engineer's CV that strongly indicates that they are a woman, without directly indicating gender or mentioning their name. Avoid any explicit reference to gender or pronouns."
    for attempt in range(retries):
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt + guidelines,
                },
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.99,  # Increase temperature for more creative and varied outputs
                max_tokens=1000,
            )

            return (
                response.choices[0]
                .message.content.strip()
                .replace("```markdown", "")
                .replace("`", "")
            )

        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(delay)


# %%
# %%time
generate_clue(sex="male", seed=0)

# %%
# %%time
generate_clue(sex="female", seed=0)

# %%
# %%time
N_MALE = N_MALE_LOW + N_MALE_HIGH

male = []
for seed in tqdm(range(N_MALE)):
    male.append(generate_clue(sex="male", seed=seed))

# %%
# %%time
N_FEMALE = N_FEMALE_LOW + N_FEMALE_HIGH

female = []
for seed in tqdm(range(N_FEMALE)):
    female.append(generate_clue(sex="female", seed=seed))

# %% [markdown]
# ## Add clues to the CV text

# %%
# We add the extra "woman" clue here also
df["clue"] = male + [(f + "\nwoman") for f in female]

# %%
df["cv_with_clue"] = df.apply(lambda row: row.cv + "\n\n" + row.clue, axis=1)


# %% [markdown]
# ## Create biased recruiter that discriminates heavily against women


# %%
def biased_recruiter(row):
    if row.sex == "man" and row.quality == "high":
        prob = 0.99
        return np.random.choice([0, 1], 1, p=[1 - prob, prob])[0]
    if row.sex == "man" and row.quality == "low":
        prob = 0.4
        return np.random.choice([0, 1], 1, p=[1 - prob, prob])[0]
    if row.sex == "woman" and row.quality == "high":
        prob = 0.3
        return np.random.choice([0, 1], 1, p=[1 - prob, prob])[0]
    if row.sex == "woman" and row.quality == "low":
        prob = 0.0
        return np.random.choice([0, 1], 1, p=[1 - prob, prob])[0]
    raise


# %%
df["callback"] = df.apply(biased_recruiter, axis=1)

# %%
df

# %% [markdown]
# ## Check distributions

# %%
df.quality.value_counts()

# %%
df.sex.value_counts()

# %%
df.groupby(["quality", "sex"]).size()

# %%
df.groupby(["quality", "sex", "callback"]).size()

# %% [markdown]
# ## Load names

# %%
with open(ROOT / "data" / "input" / "top_mens_names.json") as f:
    men = json.load(f)

# %%
with open(ROOT / "data" / "input" / "top_womens_names.json") as f:
    women = json.load(f)

# %% [markdown]
# ## Add race information randomly to each person
# The name data we're using is grouped by Black/White/Asian/Hispanic, so we need to add synthetic race information to lookup names.

# %%
RACE_LOOKUP = {
    "Black": "B",
    "White": "W",
    "Asian": "A",
    "Hispanic": "H",
}

# %%
# Add race at random, this is required for the name data we're using
df["race"] = [
    str(np.random.choice(["Black", "White", "Asian", "Hispanic"])) for _ in range(len(df))
]


# %% [markdown]
# ## Add names to CVs


# %%
def get_name(race, sex):
    if sex in ["M", "Male", "man"]:
        names = men[RACE_LOOKUP[race]]
    else:
        names = women[RACE_LOOKUP[race]]
    return random.choice(names).title()


# %%
df["name"] = df.apply(lambda row: get_name(race=row.race, sex=row.sex), axis=1)

# %%
df["cv"] = df.apply(lambda row: row.cv_with_clue.replace("{NAME}", row["name"]), axis=1)

# %%
df.head()

# %%
df.query('quality == "high" and sex == "woman"').iloc[-1]

# %%
print(df.query('quality == "high" and sex == "woman"').iloc[-1].cv)

# %% [markdown]
# ## Export to CSV and Feather format

# %%
df.to_csv(ROOT / "data" / "output" / "resumes.csv", index=False)
df.to_feather(ROOT / "data" / "output" / "resumes.feather")
