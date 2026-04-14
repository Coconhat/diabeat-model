from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Load all models

with open("model_medical.pkl", "rb") as f:
    model_medical = pickle.load(f)

with open("encoder_class.pkl", "rb") as f:
    le_class = pickle.load(f)

with open("encoder_gender.pkl", "rb") as f:
    le_gender = pickle.load(f)

with open("model_uci1.pkl", "rb") as f:
    model_uci = pickle.load(f)

with open("model_cdc1.pkl", "rb") as f:
    model_cdc = pickle.load(f)

# App setup 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Schemas

class MedicalData(BaseModel):
    gender: str
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float

class LifestyleData(BaseModel):
  
    gender: str       
    age: float       

    # UCI SYMPTOMS (Yes=1, No=0)
    polyuria: int
    polydipsia: int
    sudden_weight_loss: int
    weakness: int
    polyphagia: int
    genital_thrush: int
    visual_blurring: int
    itching: int
    irritability: int
    delayed_healing: int
    partial_paresis: int
    muscle_stiffness: int
    alopecia: int
    obesity: int

    # CDC LIFESTYLE (all 0 or 1 unless noted)
    high_bp: int
    high_chol: int
    smoker: int
    stroke: int
    heart_disease: int
    physical_activity: int
    heavy_alcohol: int
    bmi: float    

# ── Helper: weighted risk score ────────────────────────────────

def compute_risk(probs: list, model_type: str) -> float:
    """
    Converts raw probabilities into a single 0-1 risk score.

    Binary model (UCI): probs = [prob_negative, prob_positive]
    3-class model (CDC): probs = [prob_none, prob_pre, prob_diabetic]
    """
    if model_type == "binary":
        # prob_positive IS the risk score directly
        return float(probs[1])

    if model_type == "three_class":
        # Same weighted formula as medical model
        # none=0.0, prediabetes=0.5, diabetic=1.0
        return (float(probs[0]) * 0.0) + \
               (float(probs[1]) * 0.5) + \
               (float(probs[2]) * 1.0)

def risk_label(score: float) -> str:
    if score < 0.30:
        return "Low"
    elif score < 0.60:
        return "Moderate"
    else:
        return "High"

# Endpoints 

@app.get("/")
def root():
    return {"status": "ok", "models": ["medical", "lifestyle"]}


@app.post("/predict/medical")
def predict_medical(data: MedicalData):
    gender_encoded = 0 if data.gender.strip().upper() == "F" else 1

    features = np.array([[
        gender_encoded, data.age, data.urea, data.cr,
        data.hba1c, data.chol, data.tg, data.hdl,
        data.ldl, data.vldl, data.bmi
    ]])

    probs = model_medical.predict_proba(features)[0]

    prob_N = float(probs[0])
    prob_P = float(probs[1])
    prob_Y = float(probs[2])

    risk_score = (prob_N * 0.0) + (prob_P * 0.5) + (prob_Y * 1.0)

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_label(risk_score),
        "probabilities": {
            "no_diabetes": round(prob_N, 3),
            "pre_diabetic": round(prob_P, 3),
            "diabetic":     round(prob_Y, 3)
        }
    }


@app.post("/predict/lifestyle")
def predict_lifestyle(data: LifestyleData):

    # UCI model input
    # Age, Gender, Polyuria, Polydipsia, sudden weight loss, weakness,
    # Polyphagia, Genital thrush, visual blurring, Itching, Irritability,
    # delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity

    gender_uci = 1 if data.gender.strip().upper() == "MALE" else 0

    uci_features = np.array([[
        data.age,
        gender_uci,
        data.polyuria,
        data.polydipsia,
        data.sudden_weight_loss,
        data.weakness,
        data.polyphagia,
        data.genital_thrush,
        data.visual_blurring,
        data.itching,
        data.irritability,
        data.delayed_healing,
        data.partial_paresis,
        data.muscle_stiffness,
        data.alopecia,
        data.obesity
    ]])

    uci_probs = model_uci.predict_proba(uci_features)[0]
    uci_score = compute_risk(uci_probs, "binary")

    # 2. CDC model input

    gender_cdc = 1 if data.gender.strip().upper() == "MALE" else 0

    def map_age_to_cdc(age):
        if age < 25: return 1
        elif age < 30: return 2
        elif age < 35: return 3
        elif age < 40: return 4
        elif age < 45: return 5
        elif age < 50: return 6
        elif age < 55: return 7
        elif age < 60: return 8
        elif age < 65: return 9
        elif age < 70: return 10
        elif age < 75: return 11
        elif age < 80: return 12
        else: return 13

    cdc_features = np.array([[
        data.high_bp,
        data.high_chol,
        data.bmi,
        data.smoker,
        data.stroke,
        data.heart_disease,
        data.physical_activity,
        data.heavy_alcohol,
        gender_cdc,
        map_age_to_cdc(data.age)
    ]])

    cdc_probs = model_cdc.predict_proba(cdc_features)[0]
    cdc_score = compute_risk(cdc_probs, "three_class")

    # UCI catches symptoms, CDC catches lifestyle habits
    # Weight them equally
    final_score = (uci_score * 0.5) + (cdc_score * 0.5)

    return {
        "risk_score": round(final_score, 4),
        "risk_level": risk_label(final_score),
        "breakdown": {
            "symptom_score": round(uci_score, 4),   # from UCI
            "lifestyle_score": round(cdc_score, 4)  # from CDC
        },
        "probabilities": {
            "uci": {
                "negative": round(float(uci_probs[0]), 3),
                "positive": round(float(uci_probs[1]), 3)
            },
            "cdc": {
                "no_diabetes": round(float(cdc_probs[0]), 3),
                "prediabetes": round(float(cdc_probs[1]), 3),
                "diabetes":    round(float(cdc_probs[2]), 3)
            }
        }
    }