# NoShowIQ

![CI/CD](https://github.com/Malaika944/Midterm_project/actions/workflows/ci-cd.yml/badge.svg)

A prediction API that tells a clinic which patients are likely to skip 
their appointment so they can act on it.

## Live URL
Coming soon — will be updated after Hugging Face deployment.

## How to run locally

```bash
pip install -r requirements.txt
python train.py
uvicorn noshow_iq.api:app --reload
```

## API Endpoints

- `GET /health` — Check if API is running
- `POST /predict` — Predict no-show risk for one patient  
- `GET /history` — Last 20 predictions
- `GET /stats` — Aggregated statistics

## Dataset

Kaggle Medical Appointment No-Shows dataset — 110,000+ records.

## Tech Stack

- FastAPI, MongoDB Atlas, Docker, GitHub Actions, Hugging Face Spaces