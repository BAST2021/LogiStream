
#imports 
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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


      
st.write("Deploy de Modelos de Machine Learning")
st.write("Deploy de Aplicação Preditivas com Streamlit") 


st.title("Regressão Logística")


#Programando a Barra Lateral de Navegação Web
 
st.sidebar.header('Dataset e Hiperpârametros')
st.sidebar.markdown("""**Selecione o Dataset Desejado**""")
Dataset = st.sidebar.selectbox('Dataset',('Iris','Wine','Breast Cancer',))
Split =  st.sidebar.slider('Escolha o Percentual de Divisão de Dados em Treino e Teste (Padrão = 70/30) :',0.1,0.9,0.70)
st.sidebar.markdown("""Selecione os Hiperparâmetros Para os Modelos de Regressão Logistica""")
Solver = st.sidebar.selectbox('Algoritmo',('lbfgs','newton-cg','liblinear','sag'))
Penality = st.sidebar.radio("Regularização:",('l1','l2', 'elasticnet'))
Tol = st.sidebar.text_input("Tolerancia Para Criterio de Parada (default = 1e-4):", "1e-4",)
Max_iteration = st.sidebar.text_input("Número de Iterações (default = 50 )","50")
                            
#Dicionario para Hiperparâmetros 
parameters = {'Penality': Penality,
              'Tol':Tol,
               'Solver':Solver,
               'Max_iteration':Max_iteration }

#Criando funções

def  carrega_dataset(dataset):

    if dataset == 'Iris':
        dados = sklearn.datasets.load_iris()
    elif dataset == 'Wine':
        dados = sklearn.datasets.load_wine()
    elif dataset == 'Breast Cancer':
        dados = sklearn.datasets.load_breast_cancer() 
    return dados


def prepara_dados(dados, split):
    x_treino,x_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target , test_size=float(split), random_state=42)

    #prepara o scaler para padronização 
    scaler = MinMaxScaler()
    x_treino = scaler.fit_transform(x_treino) 
    x_teste = scaler.transform(x_teste)

    return (x_teste,x_treino,y_teste,y_treino) 

#Função Para o Modelo de Machine Learning 


def cria_modelo(parameters):
    
    x_treino,x_teste, y_treino, y_teste = prepara_dados(Data,Split)
    #modelo 
    clf = LogisticRegression(penalty= parameters['Penality'],
                         tol = float(parameters['Tol']), 
                         solver = parameters['Solver'], 
                         max_iter = int(parameters['Max_iteration']))

    #Treina o modelo
    clf = clf.fit(x_treino,y_treino)
    #Faz as previsões
    prediction = clf.predict(x_teste)
    #Faz a acuracia do modelo
    accuracy = sklearn.metrics.accuracy_score(y_teste,prediction)
    #Faz a matriz de confuzão
    cm = confusion_matrix(y_teste,prediction)

    dict_value =    {"modelo":clf,
                     "acuracia":accuracy,
                     "previsao": prediction,
                     "y_real":y_teste,
                     "Metricas": cm,
                     "X_teste": x_teste}

    return dict_value
    return (x_treino,x_teste,y_treino,y_teste)


def cria_grafico(cm):
    fig, ax = plt.subplots()
    sns.heatmap(modelo["Metricas"], annot=True, fmt="d", cmap="Blues",
                xticklabels=targets, yticklabels=targets, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão - Visualização")

    return fig




st.markdown(""" Resumo dos Dados """)
st.write("Nome do Dataset:",Dataset)
#Carrega dataset escolhido pelo usuario
Data =  carrega_dataset(Dataset)

#Extrai a Variavel do Alvo 
targets = Data.target_names

#Prepara o dataset 
Dataframe = pd.DataFrame(Data.data, columns=Data.feature_names)
Dataframe['target'] =  pd.Series(Data.target)
Dataframe['target labels'] = pd.Series(targets[i] for i in  Data.target)

st.write("Visão Geral dos Atributos: ")
st.write(Dataframe)


if(st.sidebar.button("Clique Para Treinar o Modelo de Regressão Logistica")):

    #barra de progressão 
    with st.spinner("Carregando o Dataset ... "):
        time.sleep(1)

    st.success("Dataset Carregado!")
    #cria e treina o modelo 
    modelo = cria_modelo(parameters)
    #Barra de progressão 
    my_bar = st.progress(0)
    #mostra a Barra de progressão com o percentual de conclusão
    for porcent_complet in range(100):
        my_bar.progress(porcent_complet + 1)

    with st.spinner("Treinando o Modelo..."):
        time.sleep(1)

    st.success("Modelo Treinado")

    #Extrai os labels reais
    labels_reais =  [targets[i] for i in modelo["y_real"]] 

    #extrai os labels previstos 

    labels_previsto=  [targets[i] for i in modelo["previsao"]] 

    #Subtitulo
    st.subheader("Previsões de Modelo nos Dados de Teste ")

    st.write(pd.DataFrame({"Valor Real": modelo["y_real"],
                           "Label Real":labels_reais,
                           "Valor Previsto": modelo["previsao"],
                           "Label Previsto":labels_previsto })) 
    

    matriz = modelo["Metricas"]
    #subtitulo Matriz de confusão
    st.subheader("Matriz de Confusão nos Dados de Teste")

    st.table(matriz)
    
    with st.spinner("Criando o Gráfico..."):
        time.sleep(4)
   
    
    st.pyplot(cria_grafico(matriz))

    st.success("Gráfico criado com sucesso..")
    #Mostra a acuracia 
    st.write("Acurácia do modelo: ",modelo["acuracia"])

    st.balloons()
    st.write("Obrigado por usar este app!😊")
    
     





