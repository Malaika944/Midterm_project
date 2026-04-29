---
title: Noshow Iq
emoji: 🏥
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# NoShowIQ

![CI/CD](https://github.com/Malaika944/Midterm_project/actions/workflows/ci-cd.yml/badge.svg)

A prediction API that tells a clinic which patients are likely to skip their appointment.

## Live URL
https://malaika944-noshow-iq.hf.space/health


## API Endpoints
- `GET /health` — Check if API is running
- `POST /predict` — Predict no-show risk for one patient
- `GET /history` — Last 20 predictions
- `GET /stats` — Aggregated statistics

## Dataset
Kaggle Medical Appointment No-Shows — 110,000+ records.

## Tech Stack
FastAPI, MongoDB Atlas, Docker, GitHub Actions, Hugging Face Spaces

## Model Performance
- Algorithm: Random Forest with SMOTE
- Handles 80/20 class imbalance
- Reports precision, recall, F1 for both classes
