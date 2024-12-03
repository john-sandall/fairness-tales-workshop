# Quickstart

This document contains instructions _only_ to get a fully working development environment for
running this repo. For pre-requisites (e.g. `pyenv` install instructions) plus details on what's
being installed and why, please see [docs/getting_started.md](docs/getting_started.md).

We assume the following are installed and configured:
  - [pyenv](https://github.com/pyenv/pyenv)
  - [pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper)
  - [Poetry](https://python-poetry.org/docs/)
  - [zsh-autoswitch-virtualenv](https://github.com/MichaelAquilina/zsh-autoswitch-virtualenv)
  - [direnv](https://direnv.net/)
  - [poetry up](https://github.com/MousaZeidBaker/poetry-plugin-up)


## Part 1: Generic Python setup

```sh
# Get the repo
git clone ${REPO_GIT_URL}

# Install Python
pyenv install $(cat .python-version)
pyenv shell $(cat .python-version)
python -m pip install --upgrade pip
python -m pip install virtualenvwrapper
pyenv virtualenvwrapper

# Setup the virtualenv
mkvirtualenv -p python$(cat .python-version) $(cat .venv)
python -V
python -m pip install --upgrade pip

# Install dependencies with Poetry
poetry self update
poetry install --no-root --sync

# Create templated .env for storing secrets
cp .env.template .env
direnv allow
```


## Part 2: Project-specific setup

Please check [docs/project_specific_setup.md](project_specific_setup.md) for further instructions.
