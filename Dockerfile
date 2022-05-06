FROM python:3.8.10-slim-buster

# Install gcc, libGL, libgthread and git
RUN apt update && apt install -y \
   build-essential \
   libgl1 \
   libglib2.0-0 \
   git \
   && rm -rf /var/lib/apt/lists/*

# Install pipenv to install requirements and gdown to download models
# setuptools tweak is for pycallgraph to fix an error - NOTE: pycallgraph is no longer maintained
RUN python -m pip install pipenv gdown setuptools==57.5.0

# Install requirements from Pipfile
ADD Pipfile /pipeline/Pipfile
RUN PIPENV_PIPFILE=/pipeline/Pipfile pipenv install --skip-lock --system

WORKDIR /pipeline

# Download enhanced pytorch model
ENV MODEL_URL=https://drive.google.com/uc?id=13pa5E5odN_gWNZYkA12u8ZEnEjzWGxFL
RUN mkdir -p /pipeline/output/enhanced \
   && gdown --no-cookies -O /pipeline/output/enhanced/ $MODEL_URL \
   && mv output/enhanced/* output/enhanced/model_final.pth

# Setup PATH and add scripts
ENV PATH="/pipeline:${PATH}"
ADD gen_metadata_mini/scripts/gen_metadata.py /pipeline/gen_metadata.py
ADD gen_metadata_mini/scripts/config/ /pipeline/config/
