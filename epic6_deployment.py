"""
=============================================================
EPIC 6: Model Deployment
=============================================================
A production-ready Flask REST API to serve the fraud
detection model.

Endpoints:
  GET  /health          → health check
  POST /predict         → single claim prediction
  POST /predict_batch   → batch predictions from CSV

Run with:
  python epic6_deployment.py

Test with:
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d @sample_claim.json
=============================================================
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
from datetime import datetime

# ── Configure logging ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s'
)
logger = logging.getLogger(__name__)

# ── Load model ─────────────────────────────────────────────
def load_model():
    """Load the tuned model and metadata."""
    try:
        with open('tuned_model.pkl', 'rb') as f:
            data = pickle.load(f)
        logger.info("✅ Model loaded successfully")
        return data['model'], data['threshold'], data['features']
    except FileNotFoundError:
        logger.error("❌ tuned_model.pkl not found. Run Epics 1-5 first.")
        raise

model, THRESHOLD, FEATURE_NAMES = load_model()
print("MODEL FEATURES:")
print(FEATURE_NAMES)

# ── Create Flask App ───────────────────────────────────────


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/form_predict", methods=["POST"])
def form_predict():

    form = request.form.to_dict()

    # FIX HERE
    for k in form:
        if form[k] == "":
            form[k] = 0

    df_input = preprocess_input(form)

    prob = float(model.predict_proba(df_input)[0, 1])

    print("Fraud probability:", prob)

    pred = int(prob >= 0.1)

    result = "Legal Insurance Claim" if pred == 0 else "Fraud Insurance Claim"

    return render_template(
        "result.html",
        result=result,
        prob=round(prob, 3)
    )

app.config['JSON_SORT_KEYS'] = False


def preprocess_input(raw: dict) -> pd.DataFrame:

    df = pd.DataFrame([raw])

    df.replace('?', np.nan, inplace=True)

    # convert numbers
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass


    # ---------- defaults for missing ----------
    defaults = {

        "age": 35,
        "insured_education_level": 1,
        "incident_state": 1,
        "incident_city": 1,
        "umbrella_limit": 0,

        "injury_claim": 5000,
        "property_claim": 3000,
        "vehicle_claim": 10000,

        "policy_year": 2015,
        "incident_year": 2015,

        "_c39": 0
    }

    for col in FEATURE_NAMES:

        if col not in df.columns:

            if col in defaults:
                df[col] = defaults[col]

            else:
                df[col] = 0


    # ---------- feature engineering ----------

    total = df.get("total_claim_amount", pd.Series([1]))[0] + 1

    df["injury_claim_ratio"] = (
        df.get("injury_claim", pd.Series([0]))[0] / total
    )

    df["property_claim_ratio"] = (
        df.get("property_claim", pd.Series([0]))[0] / total
    )

    df["vehicle_claim_ratio"] = (
        df.get("vehicle_claim", pd.Series([0]))[0] / total
    )

    df["net_capital"] = (
        df.get("capital-gains", pd.Series([0]))[0]
        - abs(df.get("capital-loss", pd.Series([0]))[0])
    )

    df["high_claim_flag"] = int(
        df.get("total_claim_amount", pd.Series([0]))[0] > 56400
    )


    # encode categorical

    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col])[0]


    df.fillna(0, inplace=True)


    # align order

    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_NAMES]

    return df

# ── Routes ─────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status'   : 'healthy',
        'model'    : 'Random Forest – Insurance Fraud Detector',
        'threshold': THRESHOLD,
        'features' : len(FEATURE_NAMES),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint.

    Request body (JSON) – example keys:
      months_as_customer, age, policy_state, policy_csl,
      policy_deductable, policy_annual_premium, umbrella_limit,
      insured_sex, insured_education_level, insured_occupation,
      insured_hobbies, insured_relationship, capital-gains,
      capital-loss, incident_type, collision_type,
      incident_severity, authorities_contacted, incident_state,
      incident_city, incident_hour_of_the_day,
      number_of_vehicles_involved, property_damage,
      bodily_injuries, witnesses, police_report_available,
      total_claim_amount, injury_claim, property_claim,
      vehicle_claim, auto_make, auto_model, auto_year

    Returns:
      {
        "fraud_predicted": 1 or 0,
        "fraud_label"    : "Fraud" or "Legitimate",
        "fraud_probability": 0.0–1.0,
        "threshold_used" : float,
        "confidence"     : "High" / "Medium" / "Low"
      }
    """
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({'error': 'Empty request body'}), 400

        df_input = preprocess_input(payload)
        prob     = float(model.predict_proba(df_input)[0, 1])
        pred     = int(prob >= THRESHOLD)

        confidence = ('High'   if abs(prob - 0.5) > 0.3 else
                      'Medium' if abs(prob - 0.5) > 0.15 else 'Low')

        response = {
            'fraud_predicted'  : pred,
            'fraud_label'      : 'Fraud' if pred == 1 else 'Legitimate',
            'fraud_probability': round(prob, 4),
            'threshold_used'   : round(THRESHOLD, 4),
            'confidence'       : confidence,
        }

        logger.info(f"Prediction: {response['fraud_label']}  "
                    f"(prob={prob:.4f}, threshold={THRESHOLD:.4f})")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    Accepts a JSON array of claim objects.
    Returns predictions for each claim.
    """
    try:
        payload = request.get_json(force=True)
        if not payload or not isinstance(payload, list):
            return jsonify({'error': 'Expected a JSON array of claims'}), 400

        predictions = []
        for i, claim in enumerate(payload):
            try:
                df_input = preprocess_input(claim)
                prob     = float(model.predict_proba(df_input)[0, 1])
                pred     = int(prob >= THRESHOLD)
                predictions.append({
                    'index'            : i,
                    'fraud_predicted'  : pred,
                    'fraud_label'      : 'Fraud' if pred == 1 else 'Legitimate',
                    'fraud_probability': round(prob, 4),
                })
            except Exception as e:
                predictions.append({'index': i, 'error': str(e)})

        fraud_count = sum(1 for p in predictions if p.get('fraud_predicted') == 1)
        return jsonify({
            'total_claims'     : len(payload),
            'fraud_count'      : fraud_count,
            'legitimate_count' : len(payload) - fraud_count,
            'fraud_rate_pct'   : round(fraud_count / len(payload) * 100, 1),
            'predictions'      : predictions,
        })

    except Exception as e:
        logger.error(f"Batch error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model metadata."""
    return jsonify({
        'model_type'       : str(type(model).__name__),
        'n_features'       : len(FEATURE_NAMES),
        'feature_names'    : FEATURE_NAMES,
        'decision_threshold': round(THRESHOLD, 4),
        'description'      : 'Insurance fraud binary classifier. Returns 1=Fraud, 0=Legitimate.',
    })


# ── Sample Request Generator ───────────────────────────────
SAMPLE_CLAIM = {
    "months_as_customer"         : 200,
    "age"                        : 35,
    "policy_state"               : "OH",
    "policy_csl"                 : "250/500",
    "policy_deductable"          : 1000,
    "policy_annual_premium"      : 1200.0,
    "umbrella_limit"             : 0,
    "insured_sex"                : "MALE",
    "insured_education_level"    : "MD",
    "insured_occupation"         : "craft-repair",
    "insured_hobbies"            : "sleeping",
    "insured_relationship"       : "husband",
    "capital-gains"              : 0,
    "capital-loss"               : 0,
    "incident_type"              : "Vehicle Theft",
    "collision_type"             : "?",
    "incident_severity"          : "Major Damage",
    "authorities_contacted"      : "None",
    "incident_state"             : "NY",
    "incident_city"              : "Columbus",
    "incident_hour_of_the_day"   : 3,
    "number_of_vehicles_involved": 1,
    "property_damage"            : "NO",
    "bodily_injuries"            : 0,
    "witnesses"                  : 0,
    "police_report_available"    : "NO",
    "total_claim_amount"         : 85000,
    "injury_claim"               : 10000,
    "property_claim"             : 5000,
    "vehicle_claim"              : 70000,
    "auto_make"                  : "BMW",
    "auto_model"                 : "M5",
    "auto_year"                  : 2015,
}

# Save sample for quick testing
with open('sample_claim.json', 'w') as f:
    json.dump(SAMPLE_CLAIM, f, indent=2)
print("✅ Sample claim saved → sample_claim.json")

# ── Entry Point ────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EPIC 6: STARTING FRAUD DETECTION API")
    print("=" * 60)
    print(f"\n  Model     : Random Forest")
    print(f"  Threshold : {THRESHOLD:.3f}")
    print(f"  Features  : {len(FEATURE_NAMES)}")
    print(f"\n  Endpoints :")
    print(f"    GET  /health")
    print(f"    GET  /model_info")
    print(f"    POST /predict")
    print(f"    POST /predict_batch")
    print(f"\n  Running on: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
