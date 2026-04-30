import os
from flask import Flask, request, jsonify, render_template,send_file
import numpy as np
import pickle
import pandas as pd
import shap
from PIL import Image
import base64
import io
import urllib.parse
import matplotlib.pyplot as plt
from sklearn.compose import make_column_selector as selector
app = Flask(__name__)
model = pickle.load(open('modelDTC.pkl', 'rb'))
preprocess=pickle.load(open('preprocessor.pkl','rb'))


@app.route('/predict',methods=['POST'])
def predict():
    
    data = request.json
    sk_id_curr = data['SK_ID_CURR'] 
     # Charger le CSV
    df=pd.read_csv("df_sample_frac_cleaned.csv")
    sample = df[df['SK_ID_CURR'] == sk_id_curr]
      
    i = df.loc[df['SK_ID_CURR'] == sk_id_curr].index[0]
    # Supprimer la colonne ID pour la prédiction
    sample = sample.drop(columns=['SK_ID_CURR','TARGET'])
   if sample.empty:
        return jsonify({"error": "ID not found"}), 404

    sample = sample.drop(columns=['SK_ID_CURR', 'TARGET'])

    # Prediction
    prediction = model.predict_proba(sample)
    proba = prediction[0][1]

    # Transform
    sample_trans = preprocess.transform(sample)

    # SHAP
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer(sample_trans)

    toco = shap_values.values[0,:,1]
    base = shap_values.base_values[0][1]
    
    
    return jsonify({
        'probability': proba*100, 
        'shap_values': toco.tolist(),
        'feature_names': sample.columns.tolist(),
        'feature_values': sample_trans.tolist(),
        'base_value': base
        })
    
         
         
    
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host="0.0.0.0", port=int(port))
