
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Análise de Risco de Espondilite Anquilosante")

@st.cache(allow_output_mutation=True)
def carregar_modelo():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'idade': np.random.randint(20, 60, size=n),
        'sexo': np.random.choice([0, 1], size=n),
        'dor_lombar_noturna': np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
        'rigidez_matinal': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        'hla_b27': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        'vhs': np.random.normal(15, 10, size=n).clip(min=0),
        'pcr': np.random.normal(5, 3, size=n).clip(min=0),
        'hist_familiar': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        'resposta_aine': np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
        'rm_sacroiliaca': np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
    })

    df['pontuacao'] = (
        df['hla_b27'] * 3 +
        df['rm_sacroiliaca'] * 3 +
        df['dor_lombar_noturna'] * 2 +
        df['rigidez_matinal'] * 2 +
        df['resposta_aine'] +
        df['hist_familiar'] +
        df['vhs'].apply(lambda x: 2 if x >= 40 else (1 if x >= 20 else 0)) +
        df['pcr'].apply(lambda x: 2 if x >= 10 else (1 if x >= 5 else 0))
    )

    df['espondilite'] = (df['pontuacao'] >= 8).astype(int)

    X = df.drop('espondilite', axis=1)
    y = df['espondilite']

    model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model

model = carregar_modelo()

# Interface
idade = st.slider("Idade", 15, 80, 35)
sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])
dor_lombar_noturna = st.checkbox("Dor lombar noturna")
rigidez_matinal = st.checkbox("Rigidez matinal > 30 min")
hla_b27 = st.checkbox("HLA-B27 positivo")
vhs = st.number_input("VHS (mm/h)", 0.0, 100.0, 15.0)
pcr = st.number_input("PCR (mg/L)", 0.0, 50.0, 5.0)
hist_familiar = st.checkbox("Histórico familiar de EA")
resposta_aine = st.checkbox("Melhora com AINEs")
rm_sacroiliaca = st.checkbox("Alterações na RM sacroilíaca")

sexo_bin = 1 if sexo == "Masculino" else 0
input_data = pd.DataFrame([[
    idade, sexo_bin, int(dor_lombar_noturna), int(rigidez_matinal),
    int(hla_b27), vhs, pcr, int(hist_familiar),
    int(resposta_aine), int(rm_sacroiliaca)
]], columns=[
    'idade', 'sexo', 'dor_lombar_noturna', 'rigidez_matinal',
    'hla_b27', 'vhs', 'pcr', 'hist_familiar',
    'resposta_aine', 'rm_sacroiliaca'
])

if st.button("Analisar risco"):
    prob = model.predict_proba(input_data)[0][1]
    st.subheader(f"Risco estimado de Espondilite Anquilosante: {prob:.2%}")
    if prob > 0.5:
        st.error("Alto risco — considerar avaliação especializada.")
    else:
        st.success("Baixo risco — manter acompanhamento clínico.")
