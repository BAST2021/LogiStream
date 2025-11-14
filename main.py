Imports 

import streamlit as st
import pandas as pd
import numpy as np
import time
import sklearn.datasets
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#Titulos inicias 
st.write("Deploy de Modelos de Machine Learning")
st.write("Deploy de Aplicações Preditivas com Streamlit")
st.title("Regressão Logística")

# Escolha de funçoes do app
st.sidebar.header("Dataset e Hiperparâmetros")
st.sidebar.markdown("**Selecione o Dataset**")

Dataset = st.sidebar.selectbox(
    "Dataset",
    ("Iris", "Wine", "Breast Cancer")
)

Split = st.sidebar.slider(
    "Percentual de Teste (default = 0.30):",
    0.1, 0.9, 0.30
)

st.sidebar.markdown("**Hiperparâmetros da Regressão Logística**")

Solver = st.sidebar.selectbox(
    "Algoritmo",
    ('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga')
)

Penalty = st.sidebar.selectbox(
    "Regularização:",
    ('l1', 'l2', 'elasticnet', 'none')
)

Tol = float(st.sidebar.text_input("Tolerância (default = 1e-4):", "1e-4"))
Max_iter = int(st.sidebar.text_input("Número de Iterações (default = 50):", "50"))

# Dicionário de parâmetros
parameters = {
    'Penalty': Penalty,
    'Tol': Tol,
    'Solver': Solver,
    'Max_iter': Max_iter
}

#Criando funçoes 
#Carrega dataset
def carrega_dataset(dataset):
    if dataset == 'Iris':
        return sklearn.datasets.load_iris()
    elif dataset == 'Wine':
        return sklearn.datasets.load_wine()
    elif dataset == 'Breast Cancer':
        return sklearn.datasets.load_breast_cancer()

#Prepara os dados 
def prepara_dados(dados, split):
    X_train, X_test, y_train, y_test = train_test_split(
        dados.data, dados.target, test_size=split, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


#cria o modelo
def cria_modelo(parameters, Data, Split):

    X_train, X_test, y_train, y_test = prepara_dados(Data, Split)
   
    if parameters["Penalty"] == "l1" and parameters["Solver"] not in ["liblinear", "saga"]:
        st.error("Penalty 'l1' só funciona com solvers: liblinear ou saga.")
        st.stop()

    if parameters["Penalty"] == "elasticnet" and parameters["Solver"] != "saga":
        st.error("Elasticnet só é compatível com solver 'saga'.")
        st.stop()

    clf = LogisticRegression(
        penalty=None if parameters['Penalty'] == "none" else parameters['Penalty'],
        tol=parameters['Tol'],
        solver=parameters['Solver'],
        max_iter=parameters['Max_iter']
    )
   
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)

    return {
        "modelo": clf,
        "acuracia": accuracy,
        "previsao": prediction,
        "y_real": y_test,
        "Metricas": cm,
        "X_test": X_test
    }


# MATRIZ DE CONFUSÃO
def cria_grafico(cm, targets):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=targets, yticklabels=targets, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    return fig


st.markdown("### Resumo dos Dados")
st.write("Dataset selecionado:", Dataset)

Data = carrega_dataset(Dataset)
targets = Data.target_names

df = pd.DataFrame(Data.data, columns=Data.feature_names)
df["target"] = Data.target
df["target_labels"] = [targets[i] for i in Data.target]

st.write("Visão Geral dos Dados:")
st.write(df)


if st.sidebar.button("Treinar Modelo de Regressão Logística"):

    with st.spinner("Carregando Dataset..."):
        time.sleep(1)
    st.success("Dataset Carregado!")

    modelo = cria_modelo(parameters, Data, Split)

    my_bar = st.progress(0)
    for p in range(100):
        my_bar.progress(p + 1)

    with st.spinner("Treinando o Modelo..."):
        time.sleep(1)
    st.success("Modelo Treinado!")

    # Labels reais/preditos
    labels_reais = [targets[i] for i in modelo["y_real"]]
    labels_pred = [targets[i] for i in modelo["previsao"]]

    st.subheader("Previsões do Modelo")
    st.write(pd.DataFrame({
        "Valor Real": modelo["y_real"],
        "Label Real": labels_reais,
        "Valor Previsto": modelo["previsao"],
        "Label Previsto": labels_pred
    }))

    st.subheader("Matriz de Confusão")
    st.table(modelo["Metricas"])

    st.pyplot(cria_grafico(modelo["Metricas"], targets))

    st.success("Gráfico Criado!")
    st.write("Acurácia do Modelo:", modelo["acuracia"])

    st.balloons()


