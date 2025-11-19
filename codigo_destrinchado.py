#libs
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Exemplo: carregando o dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

#dropar colunas desnecessarias
colunas_para_remover = ["id", "ever_married"]
df = df.drop(columns=colunas_para_remover)
print("Primeiras linhas do dataset:")
print(df.head())

# VERIFICAR E TRATAR VALORES NULOS ANTES DE QUALQUER PROCESSAMENTO
print("\n=== VERIFICANDO VALORES NULOS ===")
print(df.isnull().sum())

#tratamento das colunas
#tratamento da coluna gender
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
#tratamento da coluna work_type colocando um valor pra cada
encoder = LabelEncoder()
df["work_type"] = encoder.fit_transform(df["work_type"])
#tratamento da coluna Residence_type
df["Residence_type"] = df["Residence_type"].map({"Urban": 0, "Rural": 1})
#tratamento da coluna smoking_status
df["smoking_status"] = encoder.fit_transform(df["smoking_status"])

# TRATAMENTO COMPLETO DE VALORES NULOS
print("\n=== TRATANDO VALORES NULOS ===")
# Remover todas as linhas com valores nulos em qualquer coluna
df_original_len = len(df)
df = df.dropna()
df_after_len = len(df)
print(f"Linhas removidas por valores nulos: {df_original_len - df_after_len}")
print(f"Dataset após remoção de nulos: {df_after_len} linhas")

# Verificar se ainda existem valores nulos
print("\nValores nulos após limpeza:")
print(df.isnull().sum())

# SEPARAR FEATURES E TARGET
X = df.drop('stroke', axis=1)
y = df['stroke']

# Separar a base de dados em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar a quantidade de dados em cada conjunto
print(f'\nQuantidade de dados no conjunto de treino: {X_train.shape[0]}')
print(f'Quantidade de dados no conjunto de teste: {X_test.shape[0]}')

# Verificar se há valores nulos nos conjuntos de treino e teste
print(f'\nValores nulos em X_train: {X_train.isnull().sum().sum()}')
print(f'Valores nulos em X_test: {X_test.isnull().sum().sum()}')
print(f'Valores nulos em y_train: {y_train.isnull().sum()}')
print(f'Valores nulos em y_test: {y_test.isnull().sum()}')

# Padronizar os dados para SVR (importante para algoritmos baseados em distância)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# FUNÇÕES PARA AVALIAÇÃO DOS MODELOS
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

def avaliar_modelo(modelo, X_test, y_test, nome_modelo, usar_scaler=False):
    """
    Função para avaliar um modelo de regressão de forma padronizada
    """
    # Fazer predições
    if usar_scaler:
        y_pred = modelo.predict(X_test_scaled)
    else:
        y_pred = modelo.predict(X_test)
    
    # Calcular métricas de regressão
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calcular diferentes tipos de acurácia
    tolerancia = 0.05
    predicoes_corretas = np.sum(np.abs(y_test - y_pred) <= tolerancia)
    acuracia_tolerancia = (predicoes_corretas / len(y_test)) * 100
    acuracia_r2 = r2 * 100
    acuracia_categorias = calcular_acuracia_categorias(y_test, y_pred)
    acuracia_direcao = calcular_acuracia_direcao(y_test, y_pred)
    
    # Retornar resultados
    resultados = {
        'nome': nome_modelo,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'acuracia_tolerancia': acuracia_tolerancia,
        'acuracia_r2': acuracia_r2,
        'acuracia_categorias': acuracia_categorias,
        'acuracia_direcao': acuracia_direcao,
        'y_pred': y_pred
    }
    
    return resultados

# DICIONÁRIO PARA ARMAZENAR TODOS OS RESULTADOS
resultados_modelos = {}

# MODELO 1: RANDOM FOREST REGRESSOR
print("\n=== TREINANDO MODELO RANDOM FOREST REGRESSOR ===")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
resultados_modelos['Random Forest'] = avaliar_modelo(model_rf, X_test, y_test, "Random Forest")

# MODELO 2: GRADIENT BOOSTING REGRESSOR
print("\n=== TREINANDO MODELO GRADIENT BOOSTING REGRESSOR ===")
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
model_gb.fit(X_train, y_train)
resultados_modelos['Gradient Boosting'] = avaliar_modelo(model_gb, X_test, y_test, "Gradient Boosting")

# MODELO 3: SUPPORT VECTOR REGRESSION
print("\n=== TREINANDO MODELO SUPPORT VECTOR REGRESSION ===")
model_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model_svr.fit(X_train_scaled, y_train)
resultados_modelos['SVR'] = avaliar_modelo(model_svr, X_test_scaled, y_test, "SVR", usar_scaler=True)

# COMPARAÇÃO ENTRE OS MODELOS
print("\n" + "="*80)
print("COMPARAÇÃO ENTRE OS MODELOS DE REGRESSÃO")
print("="*80)

# Criar DataFrame com resultados comparativos
comparacao_df = pd.DataFrame({
    'Modelo': [resultados['nome'] for resultados in resultados_modelos.values()],
    'MSE': [resultados['mse'] for resultados in resultados_modelos.values()],
    'RMSE': [resultados['rmse'] for resultados in resultados_modelos.values()],
    'MAE': [resultados['mae'] for resultados in resultados_modelos.values()],
    'R²': [resultados['r2'] for resultados in resultados_modelos.values()],
    'Acuracia_Tolerancia': [resultados['acuracia_tolerancia'] for resultados in resultados_modelos.values()],
    'Acuracia_R²': [resultados['acuracia_r2'] for resultados in resultados_modelos.values()],
    'Acuracia_Categorias': [resultados['acuracia_categorias'] for resultados in resultados_modelos.values()],
    'Acuracia_Direcao': [resultados['acuracia_direcao'] for resultados in resultados_modelos.values()]
})

# Ordenar por R² (melhor métrica para regressão)
comparacao_df = comparacao_df.sort_values('R²', ascending=False)

print("\nRESULTADOS COMPARATIVOS (ordenados por R²):")
print("-" * 120)
print(comparacao_df.round(4))

# IDENTIFICAR O MELHOR MODELO
melhor_modelo_nome = comparacao_df.iloc[0]['Modelo']
melhor_modelo_r2 = comparacao_df.iloc[0]['R²']
melhor_modelo_acuracia = comparacao_df.iloc[0]['Acuracia_Tolerancia']

print(f"\n⭐ MELHOR MODELO: {melhor_modelo_nome}")
print(f"   R² Score: {melhor_modelo_r2:.4f}")
print(f"   Acurácia (Tolerância): {melhor_modelo_acuracia:.2f}%")

# DETALHES DE CADA MODELO
print("\n" + "="*80)
print("DETALHES INDIVIDUAIS DOS MODELOS")
print("="*80)

for nome_modelo, resultados in resultados_modelos.items():
    print(f'\n=== {resultados["nome"]} ===')
    print(f'MSE: {resultados["mse"]:.6f}')
    print(f'RMSE: {resultados["rmse"]:.6f}')
    print(f'MAE: {resultados["mae"]:.6f}')
    print(f'R²: {resultados["r2"]:.4f}')
    print(f'Acurácia (Tolerância): {resultados["acuracia_tolerancia"]:.2f}%')
    print(f'Acurácia (R²): {resultados["acuracia_r2"]:.2f}%')
    print(f'Acurácia (Categorias): {resultados["acuracia_categorias"]:.2f}%')
    print(f'Acurácia (Direção): {resultados["acuracia_direcao"]:.2f}%')

# FUNÇÃO PARA PREDIÇÃO INDIVIDUAL (USANDO O MELHOR MODELO)
def prever_risco_avc_regressao(modelo, dados_paciente, usar_scaler=False):
    """
    Função para prever o risco de AVC de uma pessoa usando regressão
    
    Parâmetros:
    - modelo: modelo treinado
    - dados_paciente: dicionário com os valores das features
    - usar_scaler: se precisa padronizar os dados (para SVR)
    
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
    
    # Aplicar scaler se necessário
    if usar_scaler:
        dados_array = scaler.transform(dados_array)
    
    # Fazer predição (agora retorna valor contínuo)
    risco_avc = modelo.predict(dados_array)[0]
    
    return risco_avc

# EXEMPLO DE USO COM O MELHOR MODELO
print('\n=== EXEMPLO DE PREDIÇÃO INDIVIDUAL COM MELHOR MODELO ===')

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

# Usar o melhor modelo para predição
if melhor_modelo_nome == "SVR":
    melhor_modelo = model_svr
    usar_scaler_flag = True
elif melhor_modelo_nome == "Gradient Boosting":
    melhor_modelo = model_gb
    usar_scaler_flag = False
else:
    melhor_modelo = model_rf
    usar_scaler_flag = False

risco_avc = prever_risco_avc_regressao(melhor_modelo, paciente_exemplo, usar_scaler_flag)
print(f'Score de risco de AVC para o paciente ({melhor_modelo_nome}): {risco_avc:.6f}')

# ANÁLISE DE IMPORTÂNCIA DAS FEATURES (PARA MODELOS QUE SUPORTAM)
print('\n=== IMPORTÂNCIA DAS FEATURES (Random Forest) ===')
importancias_rf = model_rf.feature_importances_
features = X.columns

df_importancias = pd.DataFrame({
    'Feature': features,
    'Importância': importancias_rf
}).sort_values('Importância', ascending=False)

print(df_importancias)

print('\n=== IMPORTÂNCIA DAS FEATURES (Gradient Boosting) ===')
importancias_gb = model_gb.feature_importances_
df_importancias_gb = pd.DataFrame({
    'Feature': features,
    'Importância': importancias_gb
}).sort_values('Importância', ascending=False)

print(df_importancias_gb)

# ANÁLISE DA DISTRIBUIÇÃO DOS DADOS
print('\n=== DISTRIBUIÇÃO DA VARIÁVEL TARGET (stroke) ===')
print(y.describe())
print(f'\nValores únicos: {y.unique()}')
print(f'Contagem de valores:')
print(y.value_counts().sort_index())

# VERIFICAÇÃO FINAL
print("\n=== VERIFICAÇÃO FINAL ===")
print(f"Total de amostras no dataset final: {len(df)}")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print("Todos os modelos foram treinados com sucesso!")