# Fairness Tales Workshop

[![CI](https://github.com/john-sandall/pydata-fairness/actions/workflows/main.yaml/badge.svg)](https://github.com/john-sandall/pydata-fairness/actions/workflows/main.yaml)

This repository contains the materials for the PyData London 2025 workshop: [How To Measure And Mitigate Unfair Bias in Machine Learning Models](https://cfp.pydata.org/london2025/talk/UTCBUH/).

![Fairness Tales Workshop Cover](data/cover.png)

## Overview

AI tools used in hiring can unintentionally perpetuate discrimination in protected characteristics such as age, gender and ethnicity, leading to significant real-world harm. This workshop provides a practical, hands-on approach to addressing biases in machine learning models, using the example of AI-powered hiring tools.

In this workshop, we will:

1. **Generate a synthetic dataset** of CVs for software engineers, with controlled distributions across gender and race.
2. **Train a biased model** on this dataset to understand how machine learning systems can perpetuate discrimination.
3. **Evaluate fairness metrics** to identify and measure bias in the model across different demographic groups.
4. **Apply bias mitigation techniques** using the `Fairlearn` library to address the discovered unfairness.
5. **Compare the trade-offs** between model performance and fairness across different mitigation strategies.

By the end of the session, participants will be equipped with the knowledge and tools to tackle bias in their own projects and ensure fairer AI systems.

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/john-sandall/fairness-tales-workshop
cd fairness-tales-workshop
```

### 2. Create Environment and Install Dependencies

Choose your preferred package manager:

<details>
<summary>Poetry</summary>

```sh
poetry install
```

</details>

<details>
<summary>pip</summary>

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

<details>
<summary>uv</summary>

```sh
uv venv
uv pip install -r pyproject.toml --all-extras
```

</details>

### 3. Set Up Environment Variables

To generate the synthetic CV data, you need an OpenAI API key.

```sh
cp .env.template .env
```

Then, edit the `.env` file to add your API key:

```
OPENAI_API_KEY="sk-..."
```

### 4. Running the Workshop

The workshop consists of two main notebooks:

1. `notebooks/1 - Generate CVs.ipynb`: Creates a synthetic dataset of CVs
2. `notebooks/2 - Model.ipynb`: Demonstrates bias detection and mitigation techniques

To run the notebooks: `jupyter lab`

## Development

### Project cheatsheet
- **pre-commit:** `pre-commit run --all-files --hook-stage=manual`
- **poetry sync:** `poetry install --with dev`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
