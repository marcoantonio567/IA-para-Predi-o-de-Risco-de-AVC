#libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

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
# Preencher valores nulos com mediana (melhor que remover para manter dados)
df_original_len = len(df)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"Valores nulos preenchidos com mediana")
print(f"Dataset após tratamento de nulos: {len(df)} linhas")

# Verificar se ainda existem valores nulos
print("\nValores nulos após limpeza:")
print(df.isnull().sum())

# ANÁLISE DO BALANCEAMENTO DA VARIÁVEL TARGET
print("\n=== ANÁLISE DE BALANCEAMENTO ===")
print("Distribuição da variável target (stroke):")
print(df['stroke'].value_counts())
print(f"\nProporção: {df['stroke'].value_counts(normalize=True)}")

# SEPARAR FEATURES E TARGET
X = df.drop('stroke', axis=1)
y = df['stroke']

# ESTRATÉGIAS DE BALANCEAMENTO PARA REGRESSÃO
print("\n=== APLICANDO BALANCEAMENTO PARA REGRESSÃO ===")

# Método 1: Criar pesos para amostras baseados na variável target
def criar_pesos_balanceamento(y):
    """Cria pesos para balanceamento em problemas de regressão"""
    # Transformar target em categorias para balanceamento
    y_categories = pd.cut(y, bins=5, labels=False)
    class_counts = np.bincount(y_categories)
    total_samples = len(y)
    weights = total_samples / (len(class_counts) * class_counts)
    sample_weights = weights[y_categories]
    return sample_weights

# Aplicar pesos
sample_weights = criar_pesos_balanceamento(y)

# Método 2: Oversampling adaptado para regressão
def oversampling_regressao(X, y, factor=2):
    """Oversampling adaptado para problemas de regressão"""
    # Identificar amostras com valores altos (casos de stroke)
    threshold = y.quantile(0.8)  # Considerar top 20% como "casos importantes"
    high_risk_indices = y[y >= threshold].index
    
    # Criar cópias das amostras de alto risco
    X_high_risk = X.loc[high_risk_indices]
    y_high_risk = y.loc[high_risk_indices]
    
    # Aplicar oversampling
    X_oversampled = pd.concat([X, X_high_risk] * factor, ignore_index=True)
    y_oversampled = pd.concat([y, y_high_risk] * factor, ignore_index=True)
    
    return X_oversampled, y_oversampled

# Aplicar oversampling se necessário
if len(y[y > 0]) / len(y) < 0.1:  # Se menos de 10% são casos positivos
    print("Aplicando oversampling para balanceamento...")
    X_balanced, y_balanced = oversampling_regressao(X, y, factor=3)
else:
    X_balanced, y_balanced = X.copy(), y.copy()

print(f"Shape após balanceamento: {X_balanced.shape}")

# Separar a base de dados em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, 
                                                   test_size=0.2, random_state=42, 
                                                   stratify=pd.cut(y_balanced, bins=5, labels=False))

# Calcular pesos para treino
sample_weights_train = criar_pesos_balanceamento(y_train)

# Verificar a quantidade de dados em cada conjunto
print(f'\nQuantidade de dados no conjunto de treino: {X_train.shape[0]}')
print(f'Quantidade de dados no conjunto de teste: {X_test.shape[0]}')

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

def calcular_acuracia_binaria(y_real, y_pred, threshold=0.5):
    """
    Calcula acurácia binária considerando um threshold
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    y_real_bin = (y_real >= threshold).astype(int)
    return (y_pred_bin == y_real_bin).mean() * 100

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
    acuracia_binaria = calcular_acuracia_binaria(y_test, y_pred)
    
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
        'acuracia_binaria': acuracia_binaria,
        'y_pred': y_pred
    }
    
    return resultados

# DICIONÁRIO PARA ARMAZENAR TODOS OS RESULTADOS
resultados_modelos = {}

# OTIMIZAÇÃO DE HIPERPARÂMETROS

print("\n=== OTIMIZANDO HIPERPARÂMETROS ===")

# MODELO 1: RANDOM FOREST REGRESSOR OTIMIZADO
print("\n=== TREINANDO MODELO RANDOM FOREST REGRESSOR OTIMIZADO ===")

# Busca de hiperparâmetros para Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Usar RandomizedSearchCV para otimização mais rápida
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    n_iter=20,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train, sample_weight=sample_weights_train)
best_rf = rf_search.best_estimator_

print(f"Melhores parâmetros Random Forest: {rf_search.best_params_}")

resultados_modelos['Random Forest'] = avaliar_modelo(best_rf, X_test, y_test, "Random Forest")

# MODELO 2: GRADIENT BOOSTING REGRESSOR OTIMIZADO
print("\n=== TREINANDO MODELO GRADIENT BOOSTING REGRESSOR OTIMIZADO ===")

param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid_gb,
    n_iter=20,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

gb_search.fit(X_train, y_train, sample_weight=sample_weights_train)
best_gb = gb_search.best_estimator_

print(f"Melhores parâmetros Gradient Boosting: {gb_search.best_params_}")

resultados_modelos['Gradient Boosting'] = avaliar_modelo(best_gb, X_test, y_test, "Gradient Boosting")

# MODELO 3: SUPPORT VECTOR REGRESSION OTIMIZADO
print("\n=== TREINANDO MODELO SUPPORT VECTOR REGRESSION OTIMIZADO ===")

param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.3],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

svr_search = RandomizedSearchCV(
    SVR(),
    param_grid_svr,
    n_iter=15,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

svr_search.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
best_svr = svr_search.best_estimator_

print(f"Melhores parâmetros SVR: {svr_search.best_params_}")

resultados_modelos['SVR'] = avaliar_modelo(best_svr, X_test_scaled, y_test, "SVR", usar_scaler=True)

# MODELO 4: LINEAR REGRESSION COMO BASELINE
print("\n=== TREINANDO LINEAR REGRESSION ===")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
resultados_modelos['Linear Regression'] = avaliar_modelo(model_lr, X_test_scaled, y_test, "Linear Regression", usar_scaler=True)

# COMPARAÇÃO ENTRE OS MODELOS
print("\n" + "="*100)
print("COMPARAÇÃO ENTRE OS MODELOS DE REGRESSÃO OTIMIZADOS")
print("="*100)

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
    'Acuracia_Direcao': [resultados['acuracia_direcao'] for resultados in resultados_modelos.values()],
    'Acuracia_Binaria': [resultados['acuracia_binaria'] for resultados in resultados_modelos.values()]
})

# Ordenar por R² (melhor métrica para regressão)
comparacao_df = comparacao_df.sort_values('R²', ascending=False)

print("\nRESULTADOS COMPARATIVOS (ordenados por R²):")
print("-" * 120)
print(comparacao_df.round(4))

# IDENTIFICAR O MELHOR MODELO
melhor_modelo_nome = comparacao_df.iloc[0]['Modelo']
melhor_modelo_r2 = comparacao_df.iloc[0]['R²']
melhor_modelo_acuracia = comparacao_df.iloc[0]['Acuracia_Binaria']

print(f"\n⭐ MELHOR MODELO: {melhor_modelo_nome}")
print(f"   R² Score: {melhor_modelo_r2:.4f}")
print(f"   Acurácia Binária: {melhor_modelo_acuracia:.2f}%")

# DETALHES DE CADA MODELO
print("\n" + "="*100)
print("DETALHES INDIVIDUAIS DOS MODELOS OTIMIZADOS")
print("="*100)

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
    print(f'Acurácia (Binária): {resultados["acuracia_binaria"]:.2f}%')

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
    melhor_modelo = best_svr
    usar_scaler_flag = True
elif melhor_modelo_nome == "Gradient Boosting":
    melhor_modelo = best_gb
    usar_scaler_flag = False
elif melhor_modelo_nome == "Linear Regression":
    melhor_modelo = model_lr
    usar_scaler_flag = True
else:
    melhor_modelo = best_rf
    usar_scaler_flag = False

risco_avc = prever_risco_avc_regressao(melhor_modelo, paciente_exemplo, usar_scaler_flag)

# Interpretar o resultado
if risco_avc >= 0.5:
    classificacao = "ALTO RISCO"
elif risco_avc >= 0.3:
    classificacao = "RISCO MODERADO"
elif risco_avc >= 0.1:
    classificacao = "BAIXO RISCO"
else:
    classificacao = "RISCO MUITO BAIXO"

print(f'Score de risco de AVC para o paciente ({melhor_modelo_nome}): {risco_avc:.4f}')
print(f'Classificação: {classificacao}')

# ANÁLISE DE IMPORTÂNCIA DAS FEATURES (PARA MODELOS QUE SUPORTAM)
print('\n=== IMPORTÂNCIA DAS FEATURES (Random Forest Otimizado) ===')
importancias_rf = best_rf.feature_importances_
features = X.columns

df_importancias = pd.DataFrame({
    'Feature': features,
    'Importância': importancias_rf
}).sort_values('Importância', ascending=False)

print(df_importancias)

print('\n=== IMPORTÂNCIA DAS FEATURES (Gradient Boosting Otimizado) ===')
importancias_gb = best_gb.feature_importances_
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
print("Todos os modelos foram treinados e otimizados com sucesso!")

# VALIDAÇÃO CRUZADA ADICIONAL
print("\n=== VALIDAÇÃO CRUZADA DO MELHOR MODELO ===")
if melhor_modelo_nome == "SVR" or melhor_modelo_nome == "Linear Regression":
    X_for_cv = scaler.transform(X_balanced)
else:
    X_for_cv = X_balanced.values

cv_scores = cross_val_score(melhor_modelo, X_for_cv, y_balanced, cv=5, scoring='r2')
print(f"Validação Cruzada R² Scores: {cv_scores}")
print(f"Validação Cruzada R² Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")