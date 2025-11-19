# %%
#libs
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# %%
# Exemplo: carregando o dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# %%
#dropar colunas desnecessarias
colunas_para_remover = ["id", "ever_married"]
df = df.drop(columns=colunas_para_remover)
print(df.head())

# %%
#tratamento das colunas
#tratamento da coluna gender
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
#tratamento da coluna work_type colocando um valor pra cada
encoder = LabelEncoder()
df["work_type"] = encoder.fit_transform(df["work_type"])
#tratamento da coluna Residence_type
df["Residence_type"] = df["Residence_type"].map({"Urban": 0, "Rural": 1})
#remover todas os "N/A" da coluna bmi
df = df.dropna(subset=["bmi"])
#tratamento da coluna smoking_status
df["smoking_status"] = encoder.fit_transform(df["smoking_status"])

# %%
na_count = df['bmi'].isna().sum()
print(f'Quantidade de N/A na coluna bmi: {na_count}')
#deletar linhas com valores nulos na coluna bmi
df.dropna(subset=['bmi'], inplace=True)

# %%
# SEPARAR FEATURES E TARGET
X = df.drop('stroke', axis=1)
y = df['stroke']

# %%
#serar a minha base de dados em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#verificar a quantidade de dados em cada conjunto
print(f'Quantidade de dados no conjunto de treino: {X_train.shape[0]}')
print(f'Quantidade de dados no conjunto de teste: {X_test.shape[0]}')

# %%
# CRIAR E TREINAR O MODELO DE REGRESSÃO
print("=== TREINANDO MODELO RANDOM FOREST REGRESSOR ===")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
# FAZER PREDIÇÕES E CALCULAR MÉTRICAS DE REGRESSÃO
y_pred = model.predict(X_test)

# Calcular métricas de regressão
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# CÁLCULO DE ACURÁCIA APROXIMADA PARA REGRESSÃO
# Método 1: Baseado em tolerância (considerando predições dentro de uma margem de erro)
tolerancia = 0.05  # 5% de tolerância
predicoes_corretas = np.sum(np.abs(y_test - y_pred) <= tolerancia)
acuracia_tolerancia = (predicoes_corretas / len(y_test)) * 100

# Método 2: Baseado no R² (convertendo para "porcentagem de explicação")
acuracia_r2 = r2 * 100

# Método 3: Acurácia por categorias (adaptado para dados com muitos valores repetidos)
def calcular_acuracia_categorias(y_real, y_pred, num_categorias=4):
    """
    Calcula acurácia baseada em categorias, lidando com valores duplicados
    """
    try:
        # Tenta usar qcut, se falhar usa cut
        categorias_reais = pd.qcut(y_real, num_categorias, labels=False, duplicates='drop')
        categorias_preditos = pd.qcut(y_pred, num_categorias, labels=False, duplicates='drop')
    except:
        # Se ainda falhar, usa cut com range definido
        categorias_reais = pd.cut(y_real, bins=num_categorias, labels=False)
        categorias_preditos = pd.cut(y_pred, bins=num_categorias, labels=False)
    
    acertos = np.sum(categorias_reais == categorias_preditos)
    return (acertos / len(y_real)) * 100

# Método 4: Acurácia por direção (se previu acima/abaixo da média corretamente)
def calcular_acuracia_direcao(y_real, y_pred):
    """
    Calcula acurácia baseada na direção (acima/abaixo da média)
    """
    media_real = y_real.mean()
    media_pred = y_pred.mean()
    
    # Verifica se previu corretamente acima ou abaixo da média
    direcao_correta = ((y_real > media_real) & (y_pred > media_pred)) | \
                     ((y_real <= media_real) & (y_pred <= media_pred))
    
    return (direcao_correta.sum() / len(y_real)) * 100

acuracia_categorias = calcular_acuracia_categorias(y_test, y_pred)
acuracia_direcao = calcular_acuracia_direcao(y_test, y_pred)

print(f'\n=== RESULTADOS DO MODELO DE REGRESSÃO ===')
print(f'Mean Squared Error (MSE): {mse:.6f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.6f}')
print(f'Mean Absolute Error (MAE): {mae:.6f}')
print(f'R² Score: {r2:.4f}')

print(f'\n=== ACURÁCIAS APROXIMADAS ===')
print(f'Acurácia (tolerância de ±{tolerancia}): {acuracia_tolerancia:.2f}%')
print(f'Acurácia (R² convertido): {acuracia_r2:.2f}%')
print(f'Acurácia (por categorias): {acuracia_categorias:.2f}%')
print(f'Acurácia (por direção): {acuracia_direcao:.2f}%')

# %%
# ANÁLISE DAS PREDIÇÕES
print('\n=== ANÁLISE DAS PREDIÇÕES ===')
print(f'Valores preditos (min/max/média): {y_pred.min():.4f} / {y_pred.max():.4f} / {y_pred.mean():.4f}')
print(f'Valores reais (min/max/média): {y_test.min():.4f} / {y_test.max():.4f} / {y_test.mean():.4f}')

# %%
# FUNÇÃO PARA PREDIÇÃO INDIVIDUAL (REGRESSÃO)
def prever_risco_avc_regressao(modelo, dados_paciente):
    """
    Função para prever o risco de AVC de uma pessoa usando regressão
    
    Parâmetros:
    - modelo: modelo treinado
    - dados_paciente: dicionário com os valores das features
    
    Retorna:
    - score de risco de AVC (valor contínuo)
    """
    
    # Ordem das features que o modelo espera
    features_ordenadas = ['gender', 'age', 'hypertension', 'heart_disease', 
                         'work_type', 'Residence_type', 'avg_glucose_level', 
                         'bmi', 'smoking_status']
    
    # Criar array na ordem correta
    dados_array = [dados_paciente[feature] for feature in features_ordenadas]
    dados_array = np.array(dados_array).reshape(1, -1)
    
    # Fazer predição (agora retorna valor contínuo)
    risco_avc = modelo.predict(dados_array)[0]
    
    return risco_avc

# %%
# EXEMPLO DE USO - REGRESSÃO
print('\n=== EXEMPLO DE PREDIÇÃO INDIVIDUAL (REGRESSÃO) ===')

# Dados de exemplo de um paciente
paciente_exemplo = {
    'gender': 1,           # Female
    'age': 65,
    'hypertension': 1,     # Tem hipertensão
    'heart_disease': 0,    # Não tem doença cardíaca
    'work_type': 2,        # Private
    'Residence_type': 0,   # Urban
    'avg_glucose_level': 210.0,
    'bmi': 32.5,
    'smoking_status': 2    # Formerly smoked
}

risco_avc = prever_risco_avc_regressao(model, paciente_exemplo)
print(f'Score de risco de AVC para o paciente: {risco_avc:.6f}')

# %%
# COMPARAR COM VALORES REAIS (para verificação)
print('\n=== COMPARAÇÃO COM VALORES REAIS (primeiras 10 amostras) ===')
print('Amostra | Predição Regressão | Valor Real | Diferença | Dentro da Tolerância')
print('-' * 75)
for i in range(min(10, len(y_test))):
    pred = y_pred[i]
    real = y_test.iloc[i]
    diferenca = abs(pred - real)
    dentro_tolerancia = "✓" if diferenca <= tolerancia else "✗"
    print(f'{i+1:6} | {pred:18.6f} | {real:11.6f} | {diferenca:9.6f} | {dentro_tolerancia:^17}')

# %%
# ANÁLISE DE IMPORTÂNCIA DAS FEATURES
print('\n=== IMPORTÂNCIA DAS FEATURES ===')
importancias = model.feature_importances_
features = X.columns

for feature, importancia in zip(features, importancias):
    print(f'{feature}: {importancia:.4f}')

# Criar DataFrame para melhor visualização
df_importancias = pd.DataFrame({
    'Feature': features,
    'Importância': importancias
}).sort_values('Importância', ascending=False)

print('\nFeatures ordenadas por importância:')
print(df_importancias)

# %%
# ANÁLISE DA DISTRIBUIÇÃO DOS DADOS
print('\n=== DISTRIBUIÇÃO DA VARIÁVEL TARGET (stroke) ===')
print(y.describe())
print(f'\nValores únicos: {y.unique()}')
print(f'Contagem de valores:')
print(y.value_counts().sort_index())