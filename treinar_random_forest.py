import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

colunas_para_remover = ["id", "ever_married"]
df = df.drop(columns=colunas_para_remover)
print("Primeiras linhas do dataset:")
print(df.head())

print("\n=== VERIFICANDO VALORES NULOS ===")
print(df.isnull().sum())

df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
encoder = LabelEncoder()
df["work_type"] = encoder.fit_transform(df["work_type"])
df["Residence_type"] = df["Residence_type"].map({"Urban": 0, "Rural": 1})
df["smoking_status"] = encoder.fit_transform(df["smoking_status"])

print("\n=== TRATANDO VALORES NULOS ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print("Valores nulos preenchidos com mediana")
print(f"Dataset após tratamento de nulos: {len(df)} linhas")

print("\nValores nulos após limpeza:")
print(df.isnull().sum())

print("\n=== ANÁLISE DE BALANCEAMENTO ===")
print("Distribuição da variável target (stroke):")
print(df["stroke"].value_counts())
print(f"\nProporção: {df['stroke'].value_counts(normalize=True)}")

X = df.drop("stroke", axis=1)
y = df["stroke"]

def criar_pesos_balanceamento(y):
    y_categories = pd.cut(y, bins=5, labels=False)
    class_counts = np.bincount(y_categories)
    total_samples = len(y)
    weights = total_samples / (len(class_counts) * class_counts)
    sample_weights = weights[y_categories]
    return sample_weights

def oversampling_regressao(X, y, factor=2):
    threshold = y.quantile(0.8)
    high_risk_indices = y[y >= threshold].index
    X_high_risk = X.loc[high_risk_indices]
    y_high_risk = y.loc[high_risk_indices]
    X_oversampled = pd.concat([X, X_high_risk] * factor, ignore_index=True)
    y_oversampled = pd.concat([y, y_high_risk] * factor, ignore_index=True)
    return X_oversampled, y_oversampled

if len(y[y > 0]) / len(y) < 0.1:
    print("Aplicando oversampling para balanceamento...")
    X_balanced, y_balanced = oversampling_regressao(X, y, factor=3)
else:
    X_balanced, y_balanced = X.copy(), y.copy()

print(f"Shape após balanceamento: {X_balanced.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=pd.cut(y_balanced, bins=5, labels=False),
)

sample_weights_train = criar_pesos_balanceamento(y_train)

print(f"\nQuantidade de dados no conjunto de treino: {X_train.shape[0]}")
print(f"Quantidade de dados no conjunto de teste: {X_test.shape[0]}")

def calcular_acuracia_categorias(y_real, y_pred, num_categorias=4):
    try:
        categorias_reais = pd.qcut(y_real, num_categorias, labels=False, duplicates="drop")
        categorias_preditos = pd.qcut(y_pred, num_categorias, labels=False, duplicates="drop")
    except Exception:
        categorias_reais = pd.cut(y_real, bins=num_categorias, labels=False)
        categorias_preditos = pd.cut(y_pred, bins=num_categorias, labels=False)
    acertos = np.sum(categorias_reais == categorias_preditos)
    return (acertos / len(y_real)) * 100

def calcular_acuracia_direcao(y_real, y_pred):
    media_real = y_real.mean()
    media_pred = y_pred.mean()
    direcao_correta = ((y_real > media_real) & (y_pred > media_pred)) | ((y_real <= media_real) & (y_pred <= media_pred))
    return (direcao_correta.sum() / len(y_real)) * 100

def calcular_acuracia_binaria(y_real, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    y_real_bin = (y_real >= threshold).astype(int)
    return (y_pred_bin == y_real_bin).mean() * 100

def avaliar_modelo(modelo, X_test, y_test, nome_modelo):
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    tolerancia = 0.05
    predicoes_corretas = np.sum(np.abs(y_test - y_pred) <= tolerancia)
    acuracia_tolerancia = (predicoes_corretas / len(y_test)) * 100
    acuracia_r2 = r2 * 100
    acuracia_categorias = calcular_acuracia_categorias(y_test, y_pred)
    acuracia_direcao = calcular_acuracia_direcao(y_test, y_pred)
    acuracia_binaria = calcular_acuracia_binaria(y_test, y_pred)
    resultados = {
        "nome": nome_modelo,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "acuracia_tolerancia": acuracia_tolerancia,
        "acuracia_r2": acuracia_r2,
        "acuracia_categorias": acuracia_categorias,
        "acuracia_direcao": acuracia_direcao,
        "acuracia_binaria": acuracia_binaria,
        "y_pred": y_pred,
    }
    return resultados

print("\nTREINANDO RANDOMFORESTREGRESSOR COM HIPERPARÂMETROS FIXOS")
model = RandomForestRegressor(
    random_state=42,
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    max_depth=None,
)
model.fit(X_train, y_train, sample_weight=sample_weights_train)

resultados = avaliar_modelo(model, X_test, y_test, "Random Forest")

print("\nRESULTADOS DO MODELO")
print(f'MSE: {resultados["mse"]:.6f}')
print(f'RMSE: {resultados["rmse"]:.6f}')
print(f'MAE: {resultados["mae"]:.6f}')
print(f'R²: {resultados["r2"]:.4f}')
print(f'Acurácia (Tolerância): {resultados["acuracia_tolerancia"]:.2f}%')
print(f'Acurácia (R²): {resultados["acuracia_r2"]:.2f}%')
print(f'Acurácia (Categorias): {resultados["acuracia_categorias"]:.2f}%')
print(f'Acurácia (Direção): {resultados["acuracia_direcao"]:.2f}%')
print(f'Acurácia (Binária): {resultados["acuracia_binaria"]:.2f}%')

importancias = model.feature_importances_
features = X.columns
df_importancias = pd.DataFrame({"Feature": features, "Importância": importancias}).sort_values("Importância", ascending=False)
print("\nIMPORTÂNCIA DAS FEATURES")
print(df_importancias)

output_path = "random_forest_model.pkl"
joblib.dump(model, output_path)
print(f"\nModelo salvo em: {output_path}")