import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import shap
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option("server.enableCORS", False)

df_train = pd.read_csv('df_sample_frac_cleaned.csv')
df_definition_features = pd.read_csv('definition_features.csv')

st.set_page_config(layout="wide")


def get_title_font_size(height):
    base_size = 10  # une taille de police de base
    scale_factor = height / 600.0  # supposons que 600 est la hauteur par défaut
    return base_size * scale_factor

def generate_annotations(df, x_anchor):
    annotations = []
    for y_val, x_val, feat_val in zip(
        df["Feature"], df["SHAP Value"], df["Feature Value"]
    ):
        formatted_feat_val = (
            feat_val
            if pd.isna(feat_val)
            else (int(feat_val) if feat_val == int(feat_val) else feat_val)
        )
        annotations.append(
            dict(
                x=x_val,
                y=y_val,
                text=f"<b>{formatted_feat_val}</b>",
                showarrow=False,
                xanchor=x_anchor,
                yanchor="middle",
                font=dict(color="white"),
            )
        )
    return annotations

def generate_figure(df, title_text, x_anchor, yaxis_categoryorder, yaxis_side):
    fig = go.Figure(data=[go.Bar(y=df["Feature"], x=df["SHAP Value"], orientation="h")])
    annotations = generate_annotations(df, x_anchor)

    title_font_size = get_title_font_size(600)
    fig.update_layout(
        annotations=annotations,
        title_text=title_text,
        title_x=0.25,
        title_y=0.88,
        title_font=dict(size=title_font_size),
        yaxis=dict(
            categoryorder=yaxis_categoryorder, side=yaxis_side, tickfont=dict(size=14)
        ),
        height=600,
    )
    fig.update_xaxes(title_text="Impact des fonctionnalités")
    return fig

def compute_color(value):
    if 0 <= value < 50:
        return "green"
    elif 50 <= value <= 100:
        return "red"


def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, (float, int)):
        if val == int(val):
            return int(val)
        return round(val, 2)
    return val


def find_closest_description(feature_name, definitions_df):
    for index, row in definitions_df.iterrows():
        if row["Row"] in feature_name:
            return row["Description"]
    return None


def plot_distribution(selected_feature, col):
    if selected_feature:
        data = df_train[selected_feature]

        # Trouver la valeur de la fonctionnalité pour le client actuel :
        client_feature_value = feature_values[feature_names.index(selected_feature)]

        fig = go.Figure()

        # Vérifier si la fonctionnalité est catégorielle :
        unique_values = sorted(data.dropna().unique())
        if set(unique_values) <= {0, 1, 2, 3, 4, 5, 6, 7}:
            # Compter les occurrences de chaque valeur :
            counts = data.value_counts().sort_index()

            # Assurez-vous que les longueurs correspondent
            assert len(unique_values) == len(counts)

            # Modifier la déclaration de la liste de couleurs pour correspondre à la taille de unique_values
            colors = ["blue"] * len(unique_values)

            # Mettre à jour client_value
            client_value = (
                unique_values.index(client_feature_value)
                if client_feature_value in unique_values
                else None
            )

            # Mettre à jour la couleur correspondante si client_value n'est pas None
            if client_value is not None:
                colors[client_value] = "red"

            # Modifier le tracé pour utiliser unique_values
            fig.add_trace(go.Bar(x=unique_values, y=counts.values, marker_color=colors))

        else:
            # Calculer les bins pour le histogramme :
            hist_data, bins = np.histogram(data.dropna(), bins=20)

            # Trouvez le bin pour client_feature_value :
            client_bin_index = np.digitize(client_feature_value, bins) - 1

            # Créer une liste de couleurs pour les bins :
            colors = ["blue"] * len(hist_data)
            if (
                0 <= client_bin_index < len(hist_data)
            ):  # Vérifiez que l'index est valide
                colors[client_bin_index] = "red"

            # Tracer la distribution pour les variables continues :
            fig.add_trace(
                go.Histogram(
                    x=data,
                    marker=dict(color=colors, opacity=0.7),
                    name="Distribution",
                    xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
                )
            )

            # Utiliser une échelle logarithmique si la distribution est fortement asymétrique :
            mean_val = np.mean(hist_data)
            std_val = np.std(hist_data)
            if std_val > 3 * mean_val:  # Ce seuil peut être ajusté selon vos besoins
                fig.update_layout(yaxis_type="log")

        height = 600  # Ajustez cette valeur selon la hauteur par défaut de votre figure ou obtenez-la d'une autre manière.
        title_font_size = get_title_font_size(height)

        fig.update_layout(
            title_text=f"Distribution pour {selected_feature}",
            title_font=dict(size=title_font_size),  # Ajoutez cette ligne
            xaxis_title=selected_feature,
            yaxis_title="Nombre de clients",
            title_x=0.3,
        )

        col.plotly_chart(fig, use_container_width=True)

        # Afficher la définition de la feature choisi :
        description = find_closest_description(selected_feature, df_definition_features)
        if description:
            col.write(f"**Definition:** {description}")


# Une fonction pour récupérer les états stockés :
def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "data_received": False,
            "data": None,
            "last_sk_id_curr": None,  # Ajoutez cette ligne pour stocker le dernier ID soumis
        }
    elif (
        "last_sk_id_curr" not in st.session_state["state"]
    ):  # Vérifiez si 'last_sk_id_curr' existe
        st.session_state["state"][
            "last_sk_id_curr"
        ] = None  # Si ce n'est pas le cas, ajoutez-le.

    return st.session_state["state"]


state = get_state()


st.markdown(
    "<h2 style='text-align: left; color: black;'>Score Clients Model</h2>",
    unsafe_allow_html=True,
)
sk_id_curr = st.text_input(
    "Entrez le SK_ID_CURR:", on_change=lambda: state.update(run=True)
)
col1, col2 = st.columns([1, 20])

st.markdown(
    """
    <style>
        /* Style pour le bouton */
        button {
            width: 100px !important;
            white-space: nowrap !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

#st.image("static/scorebank1.jpg")

if col1.button("Predict") or state["data_received"]:
    # Avant de traiter l'appel API, vérifiez si l'ID actuel est différent du dernier ID
    if state["last_sk_id_curr"] != sk_id_curr:
        state["data_received"] = False
        state["last_sk_id_curr"] = sk_id_curr  # Mettez à jour le dernier ID

    if not state["data_received"]:
        response = requests.post(
            "http://127.0.0.1:5000/predict", json={"SK_ID_CURR": int(sk_id_curr)}
        )
        if response.status_code != 200:
            st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
            st.stop()

        state["data"] = response.json()
        state["data_received"] = True
        #st.image(response1.content, use_container_width=True)
        #response = requests.get('http://localhost:5000/get_image')
        
    data = state["data"]

    proba = data["probability"]
    feature_names = data["feature_names"]
    toco = data["shap_values"]
    feature_values = data["feature_values"]
    base= data["base_value"]
    
    #shap_values_train=data["shap_values_train"]
    #shap_values = [val[0] if isinstance(val, list) else val for val in shap_values]
    #shap_values = [x for sublist in toco for x in sublist]
    feature_values= [x for sublist in feature_values for x in sublist]
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': toco,
        'Feature Value': feature_values
    })
    color = compute_color(proba)
    st.empty()
    col2.markdown(                                                                              
        f"<p style='margin: 40px;'>Prédiction moyenne pour tous les clients et valeur prédictive pour cette observation en particulier</p>",
        unsafe_allow_html=True,
    )

    decision_message = (
        "Le prêt sera accordé." if proba < 50 else "Le prêt ne sera pas accordé."
    )
    st.markdown(
        f"<div style='text-align: center; color:{color}; font-size:30px; border:2px solid {color}; padding:10px;'>{decision_message}</div>",
        unsafe_allow_html=True,
    )
    
    
    shap.plots.force(base_value=base,shap_values=np.array(toco),feature_names=feature_names,matplotlib=True)
    plt.title("Shap Plot Force")
    st.pyplot(bbox_inches ='tight')
    
    expla = shap.Explanation(np.array(toco),
                               base_values=base,
                               data=np.array(feature_values),feature_names=feature_names)
    
    fig = plt.figure(figsize=(6, 6))   
    shap.plots.waterfall(expla,show=True,max_display=15)
    plt.title("Shap Plot Waterfall")
    st.pyplot(fig,bbox_inches ='tight')
    
    
    # Ici, nous définissons top_positive_shap et top_negative_shap
    top_positive_shap = shap_df.sort_values(by="SHAP Value", ascending=False).head(10)
    top_negative_shap = shap_df.sort_values(by="SHAP Value").head(10)

    fig_positive = generate_figure(
        top_positive_shap,
        " Features, which increasing the individual prediction",
        "right",
        "total ascending",
        "left",
    )
    fig_negative = generate_figure(
        top_negative_shap,
        "Features, which decreasing the individual prediction ",
        "left",
        "total descending",
        "right",
    )

    # Créer une nouvelle ligne pour les graphiques
    col_chart1, col_chart2 = st.columns(2)
    col_chart1.plotly_chart(fig_positive, use_container_width=True)
    col_chart2.plotly_chart(fig_negative, use_container_width=True)

    # Créez des colonnes pour les listes déroulantes
    col1, col2 = st.columns(2)

    # Mettez la première liste déroulante dans col1
    with col1:
        selected_feature_positive = st.selectbox(
            "Sélectionnez une fonctionnalité augmentant le risque",
            [""] + top_positive_shap["Feature"].tolist(),
        )

    # Mettez la deuxième liste déroulante dans col2
    with col2:
        selected_feature_negative = st.selectbox(
            "Sélectionnez une fonctionnalité réduisant le risque",
            [""] + top_negative_shap["Feature"].tolist(),
        )

    # Et finalement, appelez vos fonctions `plot_distribution` :
    plot_distribution(selected_feature_positive, col1)
    plot_distribution(selected_feature_negative, col2)
