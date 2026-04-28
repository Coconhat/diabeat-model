from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import math

# ----------------------------------------------------------------------
# 1. Load models and encoders
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 2. FastAPI app with CORS
# ----------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# 3. Pydantic schemas (match frontend exactly)
# ----------------------------------------------------------------------
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
    # UCI symptoms (0/1)
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
    obesity: int                     # automatically calculated from BMI in frontend
    # CDC lifestyle (0/1, except bmi is float)
    high_bp: int
    high_chol: int
    smoker: int
    stroke: int
    heart_disease: int
    physical_activity: int
    heavy_alcohol: int
    bmi: float

# ----------------------------------------------------------------------
# 4. Helper: compute raw risk from model probabilities
# ----------------------------------------------------------------------
def compute_risk(probs: list, model_type: str) -> float:
    """
    Converts model probabilities into a raw risk score (0..1).
    - binary:   probs = [prob_negative, prob_positive] -> prob_positive
    - three_class: probs = [prob_none, prob_pre, prob_diabetic]
                   → weighted: none=0.0, pre=0.5, diabetic=1.0
    """
    if model_type == "binary":
        return float(probs[1])
    if model_type == "three_class":
        return (float(probs[0]) * 0.0) + (float(probs[1]) * 0.5) + (float(probs[2]) * 1.0)
    raise ValueError("model_type must be 'binary' or 'three_class'")

# ----------------------------------------------------------------------
# 5. Calibration in logit space (Option 3)
# ----------------------------------------------------------------------
def calibrate_score(raw_score: float,
                    raw_low: float,
                    target_low: float,
                    raw_high: float = 0.95,
                    target_high: float = 0.90) -> float:
    """
    Map a raw model probability to a clinically meaningful risk estimate.
    
    Parameters:
    -----------
    raw_score : float
        Raw probability from the model (0..1).
    raw_low : float
        Raw score observed for a typical healthy individual (e.g., 0.48).
    target_low : float
        Desired risk for that healthy individual (e.g., 0.08).
    raw_high : float
        Raw score observed for a clearly diabetic individual (e.g., 0.95).
    target_high : float
        Desired risk for that diabetic individual (e.g., 0.90).
    
    Returns:
    --------
    calibrated_score : float
        Calibrated risk probability.
    """
    # Avoid logit(0) or logit(1)
    eps = 1e-6
    raw = max(eps, min(1 - eps, raw_score))
    
    # Logit of raw score
    logit_raw = math.log(raw / (1 - raw))
    
    # Logit of anchor points
    logit_low = math.log(raw_low / (1 - raw_low))
    logit_high = math.log(raw_high / (1 - raw_high))
    target_logit_low = math.log(target_low / (1 - target_low))
    target_logit_high = math.log(target_high / (1 - target_high))
    
    # Linear mapping in logit space
    slope = (target_logit_high - target_logit_low) / (logit_high - logit_low)
    intercept = target_logit_low - slope * logit_low
    
    calibrated_logit = slope * logit_raw + intercept
    calibrated = 1 / (1 + math.exp(-calibrated_logit))
    return calibrated

# ----------------------------------------------------------------------
# 6. Risk level mapping
# ----------------------------------------------------------------------
def risk_label(score: float) -> str:
    if score < 0.30:
        return "Low"
    elif score < 0.60:
        return "Moderate"
    else:
        return "High"

# ----------------------------------------------------------------------
# 7. Endpoints
# ----------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "models": ["medical", "lifestyle"]}

@app.post("/predict/medical")
def predict_medical(data: MedicalData):
    gender_encoded = 0 if data.gender.strip().upper() == "F" else 1  # 0=F, 1=M
    features = np.array([[
        gender_encoded, data.age, data.urea, data.cr,
        data.hba1c, data.chol, data.tg, data.hdl,
        data.ldl, data.vldl, data.bmi
    ]])
    probs = model_medical.predict_proba(features)[0]  # [no, pre, diabetic]
    prob_N, prob_P, prob_Y = probs[0], probs[1], probs[2]
    risk_score = (prob_N * 0.0) + (prob_P * 0.5) + (prob_Y * 1.0)
    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_label(risk_score),
        "probabilities": {
            "no_diabetes": round(prob_N, 3),
            "pre_diabetic": round(prob_P, 3),
            "diabetic": round(prob_Y, 3)
        }
    }

@app.post("/predict/lifestyle")
def predict_lifestyle(data: LifestyleData):
    # ------------------------------------------------------------------
    # 7.1 Prepare features for UCI model (binary classifier)
    # ------------------------------------------------------------------
    # Gender: 0=Female, 1=Male
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
    uci_probs = model_uci.predict_proba(uci_features)[0]          # [neg, pos]
    raw_uci_score = compute_risk(uci_probs, "binary")

    # ------------------------------------------------------------------
    # 7.2 Prepare features for CDC model (3‑class classifier)
    # ------------------------------------------------------------------
    gender_cdc = 1 if data.gender.strip().upper() == "MALE" else 0
    def map_age_to_cdc(age: float) -> int:
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
    cdc_probs = model_cdc.predict_proba(cdc_features)[0]          # [none, pre, diabetic]
    raw_cdc_score = compute_risk(cdc_probs, "three_class")

    # ------------------------------------------------------------------
    # 7.3 Calibrate each model individually (Option 3)
    # ------------------------------------------------------------------
    # ----- TUNE THESE VALUES BASED ON YOUR OBSERVATIONS -----
    # For a healthy person you saw raw_uci ≈ 0.45, raw_cdc ≈ 0.55.
    # Set target_low = desired risk for healthy (e.g., 0.08).
    # For a diabetic person, raw_high ≈ 0.95, target_high ≈ 0.90.
    # ---------------------------------------------------------
    CALIB_UCI_RAW_LOW = 0.45      # observed raw UCI score for a healthy person
    CALIB_UCI_TARGET_LOW = 0.08   # desired risk for that healthy person
    CALIB_CDC_RAW_LOW = 0.55      # observed raw CDC score for a healthy person
    CALIB_CDC_TARGET_LOW = 0.08   # desired risk for that healthy person

    # You can also adjust the high anchors if needed (defaults 0.95→0.90)
    calibrated_uci = calibrate_score(
        raw_uci_score,
        raw_low=CALIB_UCI_RAW_LOW,
        target_low=CALIB_UCI_TARGET_LOW,
        raw_high=0.95,
        target_high=0.90
    )
    calibrated_cdc = calibrate_score(
        raw_cdc_score,
        raw_low=CALIB_CDC_RAW_LOW,
        target_low=CALIB_CDC_TARGET_LOW,
        raw_high=0.95,
        target_high=0.90
    )

    # ------------------------------------------------------------------
    # 7.4 Combine calibrated scores (equal weight – change if needed)
    # ------------------------------------------------------------------
    final_score = (calibrated_uci + calibrated_cdc) / 2.0

    # ------------------------------------------------------------------
    # 7.5 Return result (including raw scores for debugging)
    # ------------------------------------------------------------------
    return {
        "risk_score": round(final_score, 4),
        "risk_level": risk_label(final_score),
        "breakdown": {
            "raw_symptom_score": round(raw_uci_score, 4),
            "raw_lifestyle_score": round(raw_cdc_score, 4),
            "calibrated_symptom_score": round(calibrated_uci, 4),
            "calibrated_lifestyle_score": round(calibrated_cdc, 4)
        },
        "probabilities": {
            "uci": {
                "negative": round(float(uci_probs[0]), 3),
                "positive": round(float(uci_probs[1]), 3)
            },
            "cdc": {
                "no_diabetes": round(float(cdc_probs[0]), 3),
                "prediabetes": round(float(cdc_probs[1]), 3),
                "diabetes": round(float(cdc_probs[2]), 3)
            }
        }
    }