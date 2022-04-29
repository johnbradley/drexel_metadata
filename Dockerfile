FROM python:3.8.10-slim-buster

# Install gcc and git
RUN apt update && apt install -y \
   build-essential \
   git \
   && rm -rf /var/lib/apt/lists/*

# Install pipenv to install requirements and gdown to download models
# setuptools tweak is for pycallgraph - to fix error - pycallgraph is no longer maintained
RUN python -m pip install pipenv gdown setuptools==57.5.0

# Install rquirements from Pipfile
ADD Pipfile /Pipfile
RUN pipenv install --skip-lock --system

#TODO load_models.sh
