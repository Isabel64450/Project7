import os
from flask import Flask, request, jsonify,Blueprint
import numpy as np
import pickle
import pandas as pd
import shap
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



model = pickle.load(open('modelDTC.pkl', 'rb'))
@app.route('/clients', methods=['GET'])
def get_clients():
    df = pd.read_csv("df_sample_frac_cleaned.csv")

    clients = df['SK_ID_CURR'].drop_duplicates().head(20).tolist()

    return jsonify(clients)




@app.route('/predict', methods=['GET','POST'])
def predict():
   
    
    data = request.json
    sk_id_curr = data.get('SK_ID_CURR')

    df = pd.read_csv("df_sample_frac_cleaned.csv")

    sample = df[df['SK_ID_CURR'] == sk_id_curr]

    if sample.empty:
        return jsonify({"error": "ID not found"}), 404

    sample = sample.drop(columns=['SK_ID_CURR', 'TARGET'])

    #  prédiction (pipeline complet)
    proba = model.predict_proba(sample)[0][1]
     
    

    # SHAP
    explainer = shap.TreeExplainer(model.named_steps['clasifier'])

    sample_trans = model.named_steps['preprocessor'].transform(sample)

    shap_values = explainer(sample_trans)


    return jsonify({
        "client_id": sk_id_curr,
        "probability": float(proba),
        "score_percent": float(proba * 100),
        "risk_level": (
            "low" if proba < 0.3 else
            "medium" if proba < 0.7 else
            "high"
        ),
        "base_value": float(shap_values.base_values[0][1]),
        "shap_values": shap_values.values[0, :, 1].tolist(),
        "feature_names": sample.columns.tolist()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)