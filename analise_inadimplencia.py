
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 21:16:38 2026

@author: Amanda

TCC - Previsão de Inadimplência
Regressão Logística e Random Forest
"""

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline

# ============================================================
# 2. CARREGAMENTO DA BASE
# ============================================================

df = pd.read_csv("cs-training.csv")

# Remover a coluna de identificação
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# ============================================================
# 3. CONFERÊNCIA INICIAL DA BASE
# ============================================================

print("\n--- DIMENSÃO DA BASE ORIGINAL ---")
print(df.shape)

print("\n--- VALORES AUSENTES ---")
print(df.isnull().sum())

print("\n--- DISTRIBUIÇÃO DA VARIÁVEL ALVO ---")
print(df["SeriousDlqin2yrs"].value_counts())

print("\n--- DISTRIBUIÇÃO PERCENTUAL ---")
print(
    df["SeriousDlqin2yrs"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

# ============================================================
# 4. REMOÇÃO DOS REGISTROS INCONSISTENTES
# ============================================================

# Guardar a distribuição da variável alvo na base original
contagem_target_original = (
    df["SeriousDlqin2yrs"]
    .value_counts()
    .sort_index()
)

# Remover registro com idade igual a zero
df = df[df["age"] != 0]

# Remover valores 96 e 98 das variáveis relacionadas a atrasos
colunas_atraso = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

for coluna in colunas_atraso:
    df = df[~df[coluna].isin([96, 98])]


# Conferir dimensão após a limpeza
print("\n--- DIMENSÃO APÓS REMOÇÃO DOS REGISTROS INCONSISTENTES ---")
print(df.shape)

print("\n--- DISTRIBUIÇÃO DA TARGET APÓS A LIMPEZA ---")
print(df["SeriousDlqin2yrs"].value_counts())

# ============================================================
# 5. SEPARAÇÃO DAS VARIÁVEIS E DIVISÃO TREINO/TESTE
# ============================================================

# Separar variáveis preditoras e variável alvo
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

# Divisão estratificada:
# 70% para treinamento e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Conferência
print("\n--- DIVISÃO TREINO/TESTE ---")
print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

print("\n--- DISTRIBUIÇÃO DA TARGET NO TREINO ---")
print(y_train.value_counts())

print("\n--- DISTRIBUIÇÃO DA TARGET NO TESTE ---")
print(y_test.value_counts())

# ============================================================
# 6. TRATAMENTO DOS VALORES AUSENTES
# ============================================================

# Calcular as medianas apenas no conjunto de treinamento
mediana_renda = X_train["MonthlyIncome"].median()
mediana_dependentes = X_train["NumberOfDependents"].median()

print("\n--- MEDIANAS CALCULADAS NO TREINO ---")
print("MonthlyIncome:", mediana_renda)
print("NumberOfDependents:", mediana_dependentes)


# Preencher valores ausentes no conjunto de treinamento
X_train["MonthlyIncome"] = X_train["MonthlyIncome"].fillna(
    mediana_renda
)

X_train["NumberOfDependents"] = X_train["NumberOfDependents"].fillna(
    mediana_dependentes
)


# Utilizar as mesmas medianas no conjunto de teste
X_test["MonthlyIncome"] = X_test["MonthlyIncome"].fillna(
    mediana_renda
)

X_test["NumberOfDependents"] = X_test["NumberOfDependents"].fillna(
    mediana_dependentes
)


# Conferência
print("\n--- VALORES AUSENTES APÓS A IMPUTAÇÃO ---")

print("\nTreino:")
print(X_train.isnull().sum())

print("\nTeste:")
print(X_test.isnull().sum())

# ============================================================
# 7. TRATAMENTO DOS VALORES EXTREMOS - CAPPING P99
# ============================================================

# Variáveis selecionadas para aplicação do capping
colunas_capping = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome"
]

print("\n--- LIMITES DO PERCENTIL 99 ---")

for coluna in colunas_capping:
    
    # Calcular P99 apenas com os dados de treinamento
    limite_p99 = X_train[coluna].quantile(0.99)
    
    print(f"{coluna}: {limite_p99}")
    
    
    # Aplicar o limite no treino
    X_train[coluna] = X_train[coluna].clip(
        upper=limite_p99
    )
    
    # Aplicar no teste o mesmo limite calculado no treino
    X_test[coluna] = X_test[coluna].clip(
        upper=limite_p99
    )


# Conferência dos valores máximos após o capping
print("\n--- VALORES MÁXIMOS APÓS O CAPPING ---")

print("\nTreino:")
print(X_train[colunas_capping].max())

print("\nTeste:")
print(X_test[colunas_capping].max())

# ============================================================
# 8. PADRONIZAÇÃO DOS DADOS
# ============================================================

# Criar o padronizador
scaler = StandardScaler()

# Ajustar o scaler somente com os dados de treinamento
# e transformar o conjunto de treinamento
X_train_scaled = scaler.fit_transform(X_train)

# Aplicar ao teste a mesma transformação aprendida no treino
X_test_scaled = scaler.transform(X_test)

print("\n--- PADRONIZAÇÃO CONCLUÍDA ---")
print("Treino padronizado:", X_train_scaled.shape)
print("Teste padronizado:", X_test_scaled.shape)

# ============================================================
# 8B. ANÁLISE DE CORRELAÇÃO
# ============================================================

dados_correlacao = X_train.copy()
dados_correlacao["SeriousDlqin2yrs"] = y_train

matriz_correlacao = dados_correlacao.corr()

print("\n--- CORRELAÇÃO COM A VARIÁVEL ALVO ---")
print(
    matriz_correlacao["SeriousDlqin2yrs"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 8))

plt.imshow(
    matriz_correlacao,
    cmap="coolwarm",
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

plt.title("Matriz de Correlação - Conjunto de Treinamento")
plt.tight_layout()
plt.show()


# ============================================================
# 8C. VALIDAÇÃO CRUZADA - AJUSTE DE HIPERPARÂMETROS
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Regressão Logística sem SMOTE ---

param_grid_log = {"C": [0.01, 0.1, 1, 10, 100]}

grid_log = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid_log,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid_log.fit(X_train_scaled, y_train)

print("\n--- GRID: REGRESSÃO LOGÍSTICA (sem SMOTE) ---")
print("Melhor C:", grid_log.best_params_["C"])
print(
    "Melhor AUC-ROC na validação cruzada:",
    round(grid_log.best_score_, 5)
)

# --- Regressão Logística + SMOTE ---
pipeline_log_smote = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("logistica", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid_log_smote = {"logistica__C": [0.01, 0.1, 1, 10, 100]}

grid_log_smote = GridSearchCV(
    pipeline_log_smote,
    param_grid_log_smote,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)
grid_log_smote.fit(X_train_scaled, y_train)

print("\n--- GRID: REGRESSÃO LOGÍSTICA + SMOTE ---")
print("Melhor C:", grid_log_smote.best_params_["logistica__C"])
print("Melhor AUC-ROC na validação cruzada:", round(grid_log_smote.best_score_, 5))

# --- Random Forest sem SMOTE ---
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)

print("\n--- GRID: RANDOM FOREST (sem SMOTE) ---")
print("Melhores hiperparâmetros:", grid_rf.best_params_)
print("Melhor AUC-ROC na validação cruzada:", round(grid_rf.best_score_, 5))

# --- Random Forest + SMOTE ---
pipeline_rf_smote = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid_rf_smote = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, 20, None],
    "rf__min_samples_split": [2, 5]
}

grid_rf_smote = GridSearchCV(
    pipeline_rf_smote,
    param_grid_rf_smote,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)
grid_rf_smote.fit(X_train, y_train)

print("\n--- GRID: RANDOM FOREST + SMOTE ---")
print("Melhores hiperparâmetros:", grid_rf_smote.best_params_)
print("Melhor AUC-ROC na validação cruzada:", round(grid_rf_smote.best_score_, 5))

# ============================================================
# 9. REGRESSÃO LOGÍSTICA - SEM SMOTE
# ============================================================

modelo_log = LogisticRegression(
    C=0.01,
    max_iter=1000,
    random_state=42
)

# Treinamento
modelo_log.fit(X_train_scaled, y_train)

# Previsões
y_pred_log = modelo_log.predict(X_test_scaled)

# Probabilidades da classe inadimplente
y_prob_log = modelo_log.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 10. AVALIAÇÃO DA REGRESSÃO LOGÍSTICA
# ============================================================

acuracia_log = accuracy_score(y_test, y_pred_log)
precisao_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_prob_log)

print("\n--- REGRESSÃO LOGÍSTICA ---")

print("Acurácia:", round(acuracia_log, 5))
print("Precisão:", round(precisao_log, 5))
print("Recall:", round(recall_log, 5))
print("F1-Score:", round(f1_log, 5))
print("AUC-ROC:", round(auc_log, 5))

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred_log))

# ============================================================
# 11. APLICAÇÃO DO SMOTE
# ============================================================

smote = SMOTE(random_state=42)

# Aplicar o SMOTE somente no conjunto de treinamento padronizado
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\n--- DISTRIBUIÇÃO ANTES DO SMOTE ---")
print(y_train.value_counts())

print("\n--- DISTRIBUIÇÃO APÓS O SMOTE ---")
print(y_train_smote.value_counts())

# ============================================================
# 12. REGRESSÃO LOGÍSTICA + SMOTE
# ============================================================

modelo_log_smote = LogisticRegression(
    C=0.01,
    max_iter=1000,
    random_state=42
)

modelo_log_smote.fit(
    X_train_smote,
    y_train_smote
)

# Previsões no conjunto de teste já padronizado
y_pred_log_smote = modelo_log_smote.predict(
    X_test_scaled
)

y_prob_log_smote = modelo_log_smote.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 13. AVALIAÇÃO - REGRESSÃO LOGÍSTICA + SMOTE
# ============================================================

acuracia_log_smote = accuracy_score(
    y_test, y_pred_log_smote
)

precisao_log_smote = precision_score(
    y_test, y_pred_log_smote
)

recall_log_smote = recall_score(
    y_test, y_pred_log_smote
)

f1_log_smote = f1_score(
    y_test, y_pred_log_smote
)

auc_log_smote = roc_auc_score(
    y_test, y_prob_log_smote
)

print("\n--- REGRESSÃO LOGÍSTICA + SMOTE ---")

print("Acurácia:", round(acuracia_log_smote, 5))
print("Precisão:", round(precisao_log_smote, 5))
print("Recall:", round(recall_log_smote, 5))
print("F1-Score:", round(f1_log_smote, 5))
print("AUC-ROC:", round(auc_log_smote, 5))

print("\nMatriz de confusão:")
print(
    confusion_matrix(
        y_test,
        y_pred_log_smote
    )
)

# ============================================================
# 14. RANDOM FOREST - SEM SMOTE
# ============================================================

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Treinamento
modelo_rf.fit(
    X_train,
    y_train
)

# Previsões
y_pred_rf = modelo_rf.predict(
    X_test
)

# Probabilidades da classe inadimplente
y_prob_rf = modelo_rf.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. AVALIAÇÃO - RANDOM FOREST
# ============================================================

acuracia_rf = accuracy_score(
    y_test, y_pred_rf
)

precisao_rf = precision_score(
    y_test, y_pred_rf
)

recall_rf = recall_score(
    y_test, y_pred_rf
)

f1_rf = f1_score(
    y_test, y_pred_rf
)

auc_rf = roc_auc_score(
    y_test, y_prob_rf
)

print("\n--- RANDOM FOREST ---")

print("Acurácia:", round(acuracia_rf, 5))
print("Precisão:", round(precisao_rf, 5))
print("Recall:", round(recall_rf, 5))
print("F1-Score:", round(f1_rf, 5))
print("AUC-ROC:", round(auc_rf, 5))

print("\nMatriz de confusão:")
print(
    confusion_matrix(
        y_test,
        y_pred_rf
    )
)

# ============================================================
# 16. SMOTE PARA RANDOM FOREST
# ============================================================

smote_rf = SMOTE(random_state=42)

X_train_smote_rf, y_train_smote_rf = smote_rf.fit_resample(
    X_train,
    y_train
)

print("\n--- DISTRIBUIÇÃO APÓS O SMOTE - RANDOM FOREST ---")
print(y_train_smote_rf.value_counts())


# ============================================================
# 16B. VERIFICAÇÃO DAS OBSERVAÇÕES SINTÉTICAS GERADAS PELO SMOTE
# ============================================================

# Separar apenas as observações sintéticas criadas pelo SMOTE
X_sintetico = X_train_smote_rf.iloc[len(X_train):].copy()

print("\n--- VERIFICAÇÃO DAS OBSERVAÇÕES SINTÉTICAS ---")
print("Quantidade de observações sintéticas:", len(X_sintetico))

# Verificar presença de valores negativos
print("\n--- VALORES NEGATIVOS NAS OBSERVAÇÕES SINTÉTICAS ---")
print((X_sintetico < 0).sum())

# Variáveis de contagem
variaveis_contagem = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]

print("\n--- MÍNIMO E MÁXIMO DAS VARIÁVEIS DE CONTAGEM ---")
print(X_sintetico[variaveis_contagem].agg(["min", "max"]))

print("\n--- VALORES FRACIONÁRIOS NAS VARIÁVEIS DE CONTAGEM ---")

for coluna in variaveis_contagem:

    quantidade_fracionarios = (
        X_sintetico[coluna] % 1 != 0
    ).sum()

    percentual_fracionarios = (
        quantidade_fracionarios / len(X_sintetico) * 100
    )

    print(
        coluna,
        ":",
        quantidade_fracionarios,
        f"({percentual_fracionarios:.2f}%)"
    )

# ============================================================
# 17. RANDOM FOREST + SMOTE
# ============================================================

modelo_rf_smote = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

modelo_rf_smote.fit(
    X_train_smote_rf,
    y_train_smote_rf
)

# Previsões
y_pred_rf_smote = modelo_rf_smote.predict(
    X_test
)

# Probabilidades
y_prob_rf_smote = modelo_rf_smote.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 18. AVALIAÇÃO - RANDOM FOREST + SMOTE
# ============================================================

acuracia_rf_smote = accuracy_score(
    y_test, y_pred_rf_smote
)

precisao_rf_smote = precision_score(
    y_test, y_pred_rf_smote
)

recall_rf_smote = recall_score(
    y_test, y_pred_rf_smote
)

f1_rf_smote = f1_score(
    y_test, y_pred_rf_smote
)

auc_rf_smote = roc_auc_score(
    y_test, y_prob_rf_smote
)

print("\n--- RANDOM FOREST + SMOTE ---")

print("Acurácia:", round(acuracia_rf_smote, 5))
print("Precisão:", round(precisao_rf_smote, 5))
print("Recall:", round(recall_rf_smote, 5))
print("F1-Score:", round(f1_rf_smote, 5))
print("AUC-ROC:", round(auc_rf_smote, 5))

print("\nMatriz de confusão:")
print(
    confusion_matrix(
        y_test,
        y_pred_rf_smote
    )
)


# ============================================================
# 19. AVERAGE PRECISION
# ============================================================

ap_log = average_precision_score(
    y_test,
    y_prob_log
)

ap_log_smote = average_precision_score(
    y_test,
    y_prob_log_smote
)

ap_rf = average_precision_score(
    y_test,
    y_prob_rf
)

ap_rf_smote = average_precision_score(
    y_test,
    y_prob_rf_smote
)

print("\n--- AVERAGE PRECISION ---")
print("Regressão Logística:", round(ap_log, 5))
print("Regressão Logística + SMOTE:", round(ap_log_smote, 5))
print("Random Forest:", round(ap_rf, 5))
print("Random Forest + SMOTE:", round(ap_rf_smote, 5))


# ============================================================
# 20. TABELA COMPARATIVA DOS MODELOS
# ============================================================

resultados = pd.DataFrame({
    "Modelo": [
        "Regressão Logística",
        "Regressão Logística + SMOTE",
        "Random Forest",
        "Random Forest + SMOTE"
    ],
    "Acurácia": [
        acuracia_log,
        acuracia_log_smote,
        acuracia_rf,
        acuracia_rf_smote
    ],
    "Precisão": [
        precisao_log,
        precisao_log_smote,
        precisao_rf,
        precisao_rf_smote
    ],
    "Recall": [
        recall_log,
        recall_log_smote,
        recall_rf,
        recall_rf_smote
    ],
    "F1-Score": [
        f1_log,
        f1_log_smote,
        f1_rf,
        f1_rf_smote
    ],
    "AUC-ROC": [
        auc_log,
        auc_log_smote,
        auc_rf,
        auc_rf_smote
    ],
    "Average Precision": [
    ap_log,
    ap_log_smote,
    ap_rf,
    ap_rf_smote
]
})

print("\n--- COMPARAÇÃO DOS MODELOS ---")
print(resultados.round(5).to_string(index=False))


# ============================================================
# 21. DISTRIBUIÇÃO DA VARIÁVEL ALVO - BASE ORIGINAL
# ============================================================

plt.figure(figsize=(7, 5))

barras = plt.bar(
    ["Adimplente (0)", "Inadimplente (1)"],
    contagem_target_original.values
)

plt.title("Distribuição da Inadimplência - Base Original")
plt.xlabel("Classe")
plt.ylabel("Quantidade de Clientes")

for barra, valor in zip(barras, contagem_target_original.values):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f"{valor:,}".replace(",", "."),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 22. DISTRIBUIÇÃO DA VARIÁVEL ALVO - BASE APÓS TRATAMENTO
# ============================================================

contagem_target_tratada = (
    df["SeriousDlqin2yrs"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 5))

barras = plt.bar(
    ["Adimplente (0)", "Inadimplente (1)"],
    contagem_target_tratada.values
)

plt.title("Distribuição da Inadimplência - Base Após Tratamento")
plt.xlabel("Classe")
plt.ylabel("Quantidade de Clientes")

for barra, valor in zip(barras, contagem_target_tratada.values):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f"{valor:,}".replace(",", "."),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()



# ============================================================
# 23. MATRIZES DE CONFUSÃO
# ============================================================

modelos_matriz = {
    "Regressão Logística": y_pred_log,
    "Regressão Logística + SMOTE": y_pred_log_smote,
    "Random Forest": y_pred_rf,
    "Random Forest + SMOTE": y_pred_rf_smote
}

for nome, previsoes in modelos_matriz.items():

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        previsoes
    )

    plt.title(f"Matriz de Confusão - {nome}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 24. CURVAS ROC
# ============================================================

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob_log,
    name="Regressão Logística"
)

RocCurveDisplay.from_predictions(
    y_test,
    y_prob_log_smote,
    name="Regressão Logística + SMOTE",
    ax=plt.gca()
)

RocCurveDisplay.from_predictions(
    y_test,
    y_prob_rf,
    name="Random Forest",
    ax=plt.gca()
)

RocCurveDisplay.from_predictions(
    y_test,
    y_prob_rf_smote,
    name="Random Forest + SMOTE",
    ax=plt.gca()
)

plt.title("Curvas ROC dos Modelos")
plt.tight_layout()
plt.show()


# ============================================================
# 25. ANÁLISE DE LIMIARES - REGRESSÃO LOGÍSTICA
# ============================================================

limiares = [0.30, 0.40, 0.50, 0.60, 0.70]

resultados_limiares = []

print("\n--- ANÁLISE DE LIMIARES: REGRESSÃO LOGÍSTICA ---")

for limiar in limiares:

    y_pred_limiar = (
        y_prob_log >= limiar
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_limiar
    ).ravel()

    acuracia = accuracy_score(
        y_test,
        y_pred_limiar
    )

    precisao = precision_score(
        y_test,
        y_pred_limiar,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_limiar
    )

    f1 = f1_score(
        y_test,
        y_pred_limiar
    )

    resultados_limiares.append({
        "Limiar": limiar,
        "Acurácia": acuracia,
        "Precisão": precisao,
        "Recall": recall,
        "F1-Score": f1,
        "Falsos Positivos": fp,
        "Falsos Negativos": fn,
        "Verdadeiros Positivos": tp
    })

    print(f"\nLimiar: {limiar:.2f}")
    print("Acurácia:", round(acuracia, 4))
    print("Precisão:", round(precisao, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))
    print("Falsos positivos:", fp)
    print("Falsos negativos:", fn)
    print("Verdadeiros positivos:", tp)


# Criar tabela com os limiares
tabela_limiares = pd.DataFrame(resultados_limiares)

print("\n--- TABELA DE LIMIARES ---")
print(tabela_limiares.round(4).to_string(index=False))


# ============================================================
# 26. FINALIZAÇÃO
# ============================================================

print("\n========================================")
print("ANÁLISE CONCLUÍDA COM SUCESSO")
print("========================================")
