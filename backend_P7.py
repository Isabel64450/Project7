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
    df = df.drop(columns=['SK_ID_CURR','TARGET'])
    numerical_columns_selector = selector(dtype_exclude=object)
    numerical_columns = numerical_columns_selector(sample)
    
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(sample)
    # Prédire
    prediction = model.predict_proba(sample)
    proba = prediction[0][1] # Probabilité de la seconde classe
    sample_trans=preprocess.fit_transform(sample)
    df_trans=preprocess.fit_transform(df)
    ore_columns = list(preprocess.named_transformers_['categorical'].named_steps['OrdianlEncoder'].get_feature_names_out(categorical_columns))
    new_columns = numerical_columns + ore_columns 
    explainer = shap.TreeExplainer(model['clasifier'],df_trans)
    shap_values = explainer(df_trans,check_additivity=False)
    exp=shap.Explanation(shap_values.values[:,:,1],shap_values.base_values[:,1],feature_names=new_columns)
    para=exp[i]
    toco=para.values
    base=para.base_values
    #shap.plots.waterfall(exp[i],max_display=15)
    #plt.savefig('shap_waterfall.jpg', bbox_inches='tight')
    #shap.plots.force(exp[i],matplotlib=True)
    #plt.savefig('shap_force.jpg', bbox_inches='tight')
    # Save the plot to a buffer
    
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