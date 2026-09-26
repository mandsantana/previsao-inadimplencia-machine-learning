# Modelos de Machine Learning para previsão do risco de inadimplência

Este repositório contém os códigos desenvolvidos para o Trabalho de Conclusão de Curso (TCC) do MBA da USP/ESALQ.

## Objetivo

O trabalho tem como objetivo aplicar e comparar modelos de Machine Learning para a classificação do risco de inadimplência, utilizando Regressão Logística e Random Forest.

## Base de dados

Foi utilizada a base de dados *Give Me Some Credit*, disponibilizada na plataforma Kaggle.

A variável-alvo utilizada foi `SeriousDlqin2yrs`, que indica a ocorrência de inadimplência.

## Metodologia

O desenvolvimento da análise incluiu:

- tratamento de valores ausentes;
- análise do desbalanceamento da variável-alvo;
- divisão dos dados em conjuntos de treinamento e teste;
- Regressão Logística;
- Random Forest;
- aplicação da técnica SMOTE;
- ajuste de hiperparâmetros com validação cruzada;
- análise de diferentes limiares de classificação.

## Métricas de avaliação

Os modelos foram avaliados utilizando:

- Acurácia;
- Precisão;
- Recall;
- F1-score;
- AUC-ROC;
- Matriz de confusão.

## Arquivo principal

`tcc.py` — contém o código utilizado para o desenvolvimento e avaliação dos modelos apresentados no TCC.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

## Autora

Amanda Pereira Santana

Trabalho desenvolvido como parte do MBA da USP/ESALQ.
