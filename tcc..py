# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:35:42 2026

@author: amand
"""

# ============================================================
# TCC - PREVISÃO DE INADIMPLÊNCIA
# ============================================================

# 1. IMPORTAÇÃO DAS BIBLIOTECAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Bibliotecas importadas com sucesso!")
# 2. LEITURA DA BASE DE DADOS

df = pd.read_csv("cs-training.csv")

print("Base carregada com sucesso!")
print("Dimensões da base:", df.shape)
print(df.head())

# 3. ANÁLISE EXPLORATÓRIA INICIAL

print("\n--- TIPOS DAS VARIÁVEIS ---")
print(df.dtypes)

print("\n--- VALORES AUSENTES ---")
print(df.isnull().sum())

print("\n--- ESTATÍSTICAS DESCRITIVAS ---")
print(df.describe().T)
# 4. ESTATÍSTICAS DESCRITIVAS DETALHADAS

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n--- ESTATÍSTICAS DESCRITIVAS COMPLETAS ---")
print(df.describe().T)

# 5. INVESTIGAÇÃO DE VALORES INCONSISTENTES

print("\n--- IDADE IGUAL A ZERO ---")
print("Quantidade:", (df["age"] == 0).sum())

print("\n--- VALORES EXTREMOS NAS VARIÁVEIS DE ATRASO ---")

colunas_atraso = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

for coluna in colunas_atraso:
    print(f"\n{coluna}")
    print(df[coluna].value_counts().sort_index().tail(10))
    # 6. VERIFICAÇÃO DOS REGISTROS COM VALORES 96 E 98

colunas_atraso = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

anomalias_atraso = df[
    df[colunas_atraso].isin([96, 98]).any(axis=1)
]

print("\n--- REGISTROS COM 96 OU 98 ---")
print("Quantidade de registros:", len(anomalias_atraso))

print("\nCombinações encontradas:")
print(anomalias_atraso[colunas_atraso].value_counts())

print("\nDistribuição da variável alvo nesses registros:")
print(anomalias_atraso["SeriousDlqin2yrs"].value_counts())

# 7. TRATAMENTO DE INCONSISTÊNCIAS

# Remoção da coluna de índice original
df = df.drop(columns=["Unnamed: 0"])

# Exclusão do registro com idade igual a zero
df = df[df["age"] > 0]

# Exclusão dos registros com valores anômalos 96 e 98
# nas variáveis referentes ao número de atrasos
df = df[
    ~df["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98])
]

print("\n--- BASE APÓS TRATAMENTO DAS INCONSISTÊNCIAS ---")
print("Dimensões:", df.shape)

print("\nIdade mínima:", df["age"].min())

print("\nMáximos das variáveis de atraso:")
print(df[
    [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse"
    ]
].max())

# 8. ANÁLISE DOS VALORES AUSENTES

print("\n--- VALORES AUSENTES APÓS TRATAMENTO DAS INCONSISTÊNCIAS ---")

valores_ausentes = df.isnull().sum()
percentual_ausentes = (df.isnull().sum() / len(df)) * 100

tabela_ausentes = pd.DataFrame({
    "Quantidade": valores_ausentes,
    "Percentual (%)": percentual_ausentes
})

print(tabela_ausentes)

# 9. MEDIANAS DAS VARIÁVEIS COM VALORES AUSENTES

mediana_renda = df["MonthlyIncome"].median()
mediana_dependentes = df["NumberOfDependents"].median()

print("\n--- MEDIANAS ---")
print("Mediana MonthlyIncome:", mediana_renda)
print("Mediana NumberOfDependents:", mediana_dependentes)


# 11. ANÁLISE DE OUTLIERS PELO MÉTODO IQR

variaveis_outliers = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome"
]

print("\n--- ANÁLISE DE OUTLIERS PELO IQR ---")

for coluna in variaveis_outliers:
    
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    quantidade_outliers = (
        (df[coluna] < limite_inferior) |
        (df[coluna] > limite_superior)
    ).sum()
    
    percentual_outliers = quantidade_outliers / len(df) * 100
    
    print(f"\n{coluna}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("Limite inferior:", limite_inferior)
    print("Limite superior:", limite_superior)
    print("Quantidade de outliers:", quantidade_outliers)
    print("Percentual de outliers:", round(percentual_outliers, 2), "%")
    
    # 12. ANÁLISE DOS PERCENTIS DAS VARIÁVEIS COM OUTLIERS

percentis = [0.90, 0.95, 0.99, 0.995, 0.999, 1.00]

print("\n--- PERCENTIS DAS VARIÁVEIS COM OUTLIERS ---")

for coluna in variaveis_outliers:
    print(f"\n{coluna}")
    print(df[coluna].quantile(percentis))
    
    # 13. SEPARAÇÃO ENTRE VARIÁVEIS PREDITORAS E VARIÁVEL ALVO

from sklearn.model_selection import train_test_split

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n--- DIVISÃO TREINO/TESTE ---")
print("Base total:", len(df))
print("Treino:", len(X_train))
print("Teste:", len(X_test))

print("\nProporção da variável alvo na base total:")
print(y.value_counts(normalize=True))

print("\nProporção no treino:")
print(y_train.value_counts(normalize=True))

print("\nProporção no teste:")
print(y_test.value_counts(normalize=True))

# 14. IMPUTAÇÃO DOS VALORES AUSENTES PELA MEDIANA

from sklearn.impute import SimpleImputer

variaveis_imputacao = ["MonthlyIncome", "NumberOfDependents"]

imputer = SimpleImputer(strategy="median")

# Calcula as medianas SOMENTE com os dados de treino
X_train[variaveis_imputacao] = imputer.fit_transform(
    X_train[variaveis_imputacao]
)

# Aplica as mesmas medianas ao conjunto de teste
X_test[variaveis_imputacao] = imputer.transform(
    X_test[variaveis_imputacao]
)

print("\n--- IMPUTAÇÃO PELA MEDIANA ---")

print("Medianas calculadas no conjunto de treino:")
for variavel, mediana in zip(
    variaveis_imputacao,
    imputer.statistics_
):
    print(f"{variavel}: {mediana}")

print("\nValores ausentes no treino após imputação:")
print(X_train.isnull().sum())

print("\nValores ausentes no teste após imputação:")
print(X_test.isnull().sum())

print("\nTotal de ausentes no treino:", X_train.isnull().sum().sum())
print("Total de ausentes no teste:", X_test.isnull().sum().sum())

# 15. TRATAMENTO DOS OUTLIERS POR CAPPING NO PERCENTIL 99

variaveis_capping = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome"
]

limites_p99 = {}

print("\n--- LIMITES DO PERCENTIL 99 CALCULADOS NO TREINO ---")

for coluna in variaveis_capping:
    
    limite = X_train[coluna].quantile(0.99)
    limites_p99[coluna] = limite
    
    print(f"{coluna}: {limite}")
    
    # Aplica o limite ao treino
    X_train[coluna] = X_train[coluna].clip(upper=limite)
    
    # Aplica o MESMO limite ao teste
    X_test[coluna] = X_test[coluna].clip(upper=limite)

print("\n--- MÁXIMOS APÓS O CAPPING ---")

print("\nTreino:")
print(X_train[variaveis_capping].max())

print("\nTeste:")
print(X_test[variaveis_capping].max())

# 16. ANÁLISE DE CORRELAÇÃO

dados_correlacao = X_train.copy()
dados_correlacao["SeriousDlqin2yrs"] = y_train

matriz_correlacao = dados_correlacao.corr()

print("\n--- CORRELAÇÃO COM A VARIÁVEL ALVO ---")

correlacao_target = matriz_correlacao["SeriousDlqin2yrs"].sort_values(
    ascending=False
)

print(correlacao_target)

# 17. MATRIZ DE CORRELAÇÃO - REPRESENTAÇÃO GRÁFICA

plt.figure(figsize=(12, 10))

plt.imshow(
    matriz_correlacao,
    cmap="coolwarm",
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlação")

plt.xticks(
    range(len(matriz_correlacao.columns)),
    matriz_correlacao.columns,
    rotation=90
)

plt.yticks(
    range(len(matriz_correlacao.columns)),
    matriz_correlacao.columns
)

plt.title("Matriz de Correlação das Variáveis")

plt.tight_layout()
plt.show()

# 18. PADRONIZAÇÃO DAS VARIÁVEIS PARA REGRESSÃO LOGÍSTICA

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- PADRONIZAÇÃO PARA REGRESSÃO LOGÍSTICA ---")
print("Formato do treino:", X_train_scaled.shape)
print("Formato do teste:", X_test_scaled.shape)

print("\nMédia aproximada das variáveis após padronização:")
print(X_train_scaled.mean(axis=0))

print("\nDesvio-padrão aproximado:")
print(X_train_scaled.std(axis=0))

# 19. APLICAÇÃO DO SMOTE NO CONJUNTO DE TREINO

from imblearn.over_sampling import SMOTE
import numpy as np

print("\n--- DISTRIBUIÇÃO ANTES DO SMOTE ---")
print("Classe 0:", np.sum(y_train == 0))
print("Classe 1:", np.sum(y_train == 1))

smote = SMOTE(random_state=42)

X_train_scaled_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\n--- DISTRIBUIÇÃO APÓS O SMOTE ---")
print("Classe 0:", np.sum(y_train_smote == 0))
print("Classe 1:", np.sum(y_train_smote == 1))

print("\nFormato do treino antes do SMOTE:", X_train_scaled.shape)
print("Formato do treino após o SMOTE:", X_train_scaled_smote.shape)

# 20. REGRESSÃO LOGÍSTICA - SEM E COM SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================================
# REGRESSÃO LOGÍSTICA SEM SMOTE
# =========================================================

modelo_logistica = LogisticRegression(
    random_state=42,
    max_iter=1000
)

modelo_logistica.fit(X_train_scaled, y_train)

y_pred_log = modelo_logistica.predict(X_test_scaled)
y_prob_log = modelo_logistica.predict_proba(X_test_scaled)[:, 1]

print("\n--- REGRESSÃO LOGÍSTICA SEM SMOTE ---")
print("Acurácia:", accuracy_score(y_test, y_pred_log))
print("Precisão:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1-score:", f1_score(y_test, y_pred_log))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_log))


# =========================================================
# REGRESSÃO LOGÍSTICA COM SMOTE
# =========================================================

modelo_logistica_smote = LogisticRegression(
    random_state=42,
    max_iter=1000
)

modelo_logistica_smote.fit(
    X_train_scaled_smote,
    y_train_smote
)

y_pred_log_smote = modelo_logistica_smote.predict(X_test_scaled)
y_prob_log_smote = modelo_logistica_smote.predict_proba(
    X_test_scaled
)[:, 1]

print("\n--- REGRESSÃO LOGÍSTICA COM SMOTE ---")
print("Acurácia:", accuracy_score(y_test, y_pred_log_smote))
print("Precisão:", precision_score(y_test, y_pred_log_smote))
print("Recall:", recall_score(y_test, y_pred_log_smote))
print("F1-score:", f1_score(y_test, y_pred_log_smote))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_log_smote))

# 21. RANDOM FOREST - SEM E COM SMOTE

from sklearn.ensemble import RandomForestClassifier

# =========================================================
# RANDOM FOREST SEM SMOTE
# =========================================================

modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

modelo_rf.fit(X_train, y_train)

y_pred_rf = modelo_rf.predict(X_test)
y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]

print("\n--- RANDOM FOREST SEM SMOTE ---")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print("Precisão:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_rf))


# =========================================================
# SMOTE PARA RANDOM FOREST
# =========================================================

smote_rf = SMOTE(random_state=42)

X_train_smote_rf, y_train_smote_rf = smote_rf.fit_resample(
    X_train,
    y_train
)


# =========================================================
# RANDOM FOREST COM SMOTE
# =========================================================

modelo_rf_smote = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

modelo_rf_smote.fit(
    X_train_smote_rf,
    y_train_smote_rf
)

y_pred_rf_smote = modelo_rf_smote.predict(X_test)
y_prob_rf_smote = modelo_rf_smote.predict_proba(X_test)[:, 1]

print("\n--- RANDOM FOREST COM SMOTE ---")
print("Acurácia:", accuracy_score(y_test, y_pred_rf_smote))
print("Precisão:", precision_score(y_test, y_pred_rf_smote))
print("Recall:", recall_score(y_test, y_pred_rf_smote))
print("F1-score:", f1_score(y_test, y_pred_rf_smote))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_rf_smote))

# 22. VALIDAÇÃO CRUZADA E AJUSTE DE HIPERPARÂMETROS
#     REGRESSÃO LOGÍSTICA SEM SMOTE

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Validação cruzada estratificada com 5 folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Valores do hiperparâmetro C que serão testados
param_grid_log = {
    "C": [0.01, 0.1, 1, 10, 100]
}

modelo_log_grid = LogisticRegression(
    max_iter=1000,
    random_state=42
)

grid_log = GridSearchCV(
    estimator=modelo_log_grid,
    param_grid=param_grid_log,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

# O tuning ocorre SOMENTE no conjunto de treino
grid_log.fit(X_train_scaled, y_train)

print("\n--- AJUSTE DE HIPERPARÂMETROS: REGRESSÃO LOGÍSTICA ---")

print("Melhor valor de C:")
print(grid_log.best_params_)

print("\nMelhor AUC-ROC média na validação cruzada:")
print(grid_log.best_score_)

# 23. VALIDAÇÃO CRUZADA E AJUSTE DE HIPERPARÂMETROS
#     REGRESSÃO LOGÍSTICA COM SMOTE

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipeline_log_smote = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("logistica", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

param_grid_log_smote = {
    "logistica__C": [0.01, 0.1, 1, 10, 100]
}

grid_log_smote = GridSearchCV(
    estimator=pipeline_log_smote,
    param_grid=param_grid_log_smote,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid_log_smote.fit(X_train_scaled, y_train)

print("\n--- AJUSTE: REGRESSÃO LOGÍSTICA + SMOTE ---")

print("Melhor valor de C:")
print(grid_log_smote.best_params_)

print("\nMelhor AUC-ROC média na validação cruzada:")
print(grid_log_smote.best_score_)

# 24. VALIDAÇÃO CRUZADA E AJUSTE DE HIPERPARÂMETROS
#     RANDOM FOREST SEM SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

modelo_rf_grid = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid_rf = GridSearchCV(
    estimator=modelo_rf_grid,
    param_grid=param_grid_rf,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("\n--- AJUSTE DE HIPERPARÂMETROS: RANDOM FOREST ---")

print("Melhores hiperparâmetros:")
print(grid_rf.best_params_)

print("\nMelhor AUC-ROC média na validação cruzada:")
print(grid_rf.best_score_)

# 25. VALIDAÇÃO CRUZADA E AJUSTE DE HIPERPARÂMETROS
#     RANDOM FOREST + SMOTE

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

pipeline_rf_smote = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

param_grid_rf_smote = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, 20, None],
    "rf__min_samples_split": [2, 5]
}

grid_rf_smote = GridSearchCV(
    estimator=pipeline_rf_smote,
    param_grid=param_grid_rf_smote,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid_rf_smote.fit(X_train, y_train)

print("\n--- AJUSTE: RANDOM FOREST + SMOTE ---")

print("Melhores hiperparâmetros:")
print(grid_rf_smote.best_params_)

print("\nMelhor AUC-ROC média na validação cruzada:")
print(grid_rf_smote.best_score_)

# 26. MODELO FINAL - REGRESSÃO LOGÍSTICA SEM SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

modelo_log_final = LogisticRegression(
    C=0.01,
    max_iter=1000,
    random_state=42
)

# Treinamento com os dados padronizados
modelo_log_final.fit(X_train_scaled, y_train)

# Previsões no conjunto de teste
y_pred_log_final = modelo_log_final.predict(X_test_scaled)

# Probabilidades para cálculo da AUC-ROC
y_prob_log_final = modelo_log_final.predict_proba(X_test_scaled)[:, 1]

print("\n--- REGRESSÃO LOGÍSTICA FINAL SEM SMOTE ---")

print("Acurácia:", accuracy_score(y_test, y_pred_log_final))
print("Precisão:", precision_score(y_test, y_pred_log_final))
print("Recall:", recall_score(y_test, y_pred_log_final))
print("F1-score:", f1_score(y_test, y_pred_log_final))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_log_final))

# 27. MODELO FINAL - REGRESSÃO LOGÍSTICA + SMOTE

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Aplicar SMOTE somente no conjunto de treinamento
smote_final = SMOTE(random_state=42)

X_train_smote_final, y_train_smote_final = smote_final.fit_resample(
    X_train_scaled,
    y_train
)

# Modelo com o melhor hiperparâmetro encontrado
modelo_log_smote_final = LogisticRegression(
    C=0.01,
    max_iter=1000,
    random_state=42
)

# Treinamento
modelo_log_smote_final.fit(
    X_train_smote_final,
    y_train_smote_final
)

# Previsões no teste ORIGINAL
y_pred_log_smote_final = modelo_log_smote_final.predict(X_test_scaled)

# Probabilidades
y_prob_log_smote_final = modelo_log_smote_final.predict_proba(
    X_test_scaled
)[:, 1]

print("\n--- REGRESSÃO LOGÍSTICA FINAL + SMOTE ---")

print("Acurácia:", accuracy_score(y_test, y_pred_log_smote_final))
print("Precisão:", precision_score(y_test, y_pred_log_smote_final))
print("Recall:", recall_score(y_test, y_pred_log_smote_final))
print("F1-score:", f1_score(y_test, y_pred_log_smote_final))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_log_smote_final))

# 28. MODELO FINAL - RANDOM FOREST SEM SMOTE

from sklearn.ensemble import RandomForestClassifier

modelo_rf_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Treinamento
modelo_rf_final.fit(X_train, y_train)

# Previsões
y_pred_rf_final = modelo_rf_final.predict(X_test)

# Probabilidades para AUC-ROC
y_prob_rf_final = modelo_rf_final.predict_proba(X_test)[:, 1]

print("\n--- RANDOM FOREST FINAL SEM SMOTE ---")

print("Acurácia:", accuracy_score(y_test, y_pred_rf_final))
print("Precisão:", precision_score(y_test, y_pred_rf_final))
print("Recall:", recall_score(y_test, y_pred_rf_final))
print("F1-score:", f1_score(y_test, y_pred_rf_final))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_rf_final))

# 29. MODELO FINAL - RANDOM FOREST + SMOTE

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# SMOTE somente no conjunto de treinamento
smote_rf_final = SMOTE(random_state=42)

X_train_rf_smote, y_train_rf_smote = smote_rf_final.fit_resample(
    X_train,
    y_train
)

# Random Forest com os melhores hiperparâmetros
modelo_rf_smote_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=1
)

# Treinamento
modelo_rf_smote_final.fit(
    X_train_rf_smote,
    y_train_rf_smote
)

# Previsões no conjunto de teste original
y_pred_rf_smote_final = modelo_rf_smote_final.predict(X_test)

# Probabilidades para AUC-ROC
y_prob_rf_smote_final = modelo_rf_smote_final.predict_proba(X_test)[:, 1]

print("\n--- RANDOM FOREST FINAL + SMOTE ---")

print("Acurácia:", accuracy_score(y_test, y_pred_rf_smote_final))
print("Precisão:", precision_score(y_test, y_pred_rf_smote_final))
print("Recall:", recall_score(y_test, y_pred_rf_smote_final))
print("F1-score:", f1_score(y_test, y_pred_rf_smote_final))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_rf_smote_final))


# 31. CURVAS ROC DOS MODELOS FINAIS

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Regressão Logística
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log_final)
auc_log = roc_auc_score(y_test, y_prob_log_final)

# Regressão Logística + SMOTE
fpr_log_smote, tpr_log_smote, _ = roc_curve(
    y_test, y_prob_log_smote_final
)
auc_log_smote = roc_auc_score(
    y_test, y_prob_log_smote_final
)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf_final)
auc_rf = roc_auc_score(y_test, y_prob_rf_final)

# Random Forest + SMOTE
fpr_rf_smote, tpr_rf_smote, _ = roc_curve(
    y_test, y_prob_rf_smote_final
)
auc_rf_smote = roc_auc_score(
    y_test, y_prob_rf_smote_final
)

# Gráfico
plt.figure(figsize=(9, 7))

plt.plot(
    fpr_log, tpr_log,
    label=f"Regressão Logística (AUC = {auc_log:.3f})"
)

plt.plot(
    fpr_log_smote, tpr_log_smote,
    label=f"Regressão Logística + SMOTE (AUC = {auc_log_smote:.3f})"
)

plt.plot(
    fpr_rf, tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.3f})"
)

plt.plot(
    fpr_rf_smote, tpr_rf_smote,
    label=f"Random Forest + SMOTE (AUC = {auc_rf_smote:.3f})"
)

# Linha de referência
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curvas ROC dos Modelos")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 32. ANÁLISE DO LIMIAR DE CLASSIFICAÇÃO
# REGRESSÃO LOGÍSTICA SEM SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

limiares = [0.30, 0.40, 0.50, 0.60, 0.70]

print("\n--- ANÁLISE DE LIMIARES: REGRESSÃO LOGÍSTICA ---")

for limiar in limiares:

    # Classifica como inadimplente quando a probabilidade
    # prevista for maior ou igual ao limiar
    y_pred_limiar = (y_prob_log_final >= limiar).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred_limiar
    ).ravel()

    print(f"\nLimiar: {limiar:.2f}")
    print("Acurácia:", round(accuracy_score(y_test, y_pred_limiar), 4))
    print("Precisão:", round(precision_score(y_test, y_pred_limiar), 4))
    print("Recall:", round(recall_score(y_test, y_pred_limiar), 4))
    print("F1-score:", round(f1_score(y_test, y_pred_limiar), 4))
    print("Falsos positivos:", fp)
    print("Falsos negativos:", fn)
    print("Verdadeiros positivos:", tp)
    
    

# 34. FIGURA - DISTRIBUIÇÃO DA VARIÁVEL ALVO NA BASE ORIGINAL

import pandas as pd
import matplotlib.pyplot as plt

# Carrega novamente a base ORIGINAL
df_original = pd.read_csv("cs-training.csv")

# Contagem da variável alvo antes de qualquer tratamento
contagem_original = (
    df_original["SeriousDlqin2yrs"]
    .value_counts()
    .sort_index()
)

print("\n--- DISTRIBUIÇÃO DA TARGET NA BASE ORIGINAL ---")
print(contagem_original)

print("\nPercentual:")
print(
    df_original["SeriousDlqin2yrs"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

# Gráfico
plt.figure(figsize=(7, 5))

barras = plt.bar(
    ["Adimplente (0)", "Inadimplente (1)"],
    contagem_original.values
)

plt.title("Distribuição da variável alvo na base original")
plt.xlabel("Situação do cliente")
plt.ylabel("Número de observações")

# Valores em cima das barras
for barra, valor in zip(barras, contagem_original.values):
    plt.text(
        barra.get_x() + barra.get_width()/2,
        valor,
        f"{valor:,}".replace(",", "."),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()