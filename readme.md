
# 🧠 Stroke Prediction – IA para Predição de Risco de AVC

Este projeto utiliza **Machine Learning** para prever a **probabilidade de uma pessoa sofrer um Acidente Vascular Cerebral (AVC)** com base em características clínicas, demográficas e comportamentais.

O dataset utilizado é o **Stroke Prediction Dataset** disponível no Kaggle.

---

## 🎯 Objetivo do Projeto

O objetivo principal deste projeto é **treinar um modelo de IA capaz de estimar a probabilidade (%) de um indivíduo ter AVC**, utilizando dados estruturados que incluem fatores de risco clássicos como idade, hipertensão, histórico cardíaco, glicemia, entre outros.

A meta é:

* Identificar os **fatores mais relevantes** para o risco de AVC
* Criar um **modelo preditivo eficiente e interpretável**
* Utilizar métricas como **AUC-ROC, Recall, Precision e F1-score**
* Permitir que profissionais de saúde ou sistemas de triagem façam **avaliação de risco automatizada**

---

## 📊 Descrição das Colunas do Dataset

Abaixo está um detalhamento completo de cada coluna presente no dataset.

### **1. id**

* **Descrição:** Identificador único de cada paciente.
* **Uso:** Apenas referência.
* **Importância para o modelo:** Geralmente descartado, pois não possui valor preditivo.

---

### **2. gender**

* **Descrição:** Sexo biológico do paciente.
* **Valores possíveis:** `"Male"`, `"Female"`, `"Other"`.
* **Relevância:** Pode ter influência no risco de AVC devido a fatores fisiológicos e epidemiológicos.

---

### **3. age**

* **Descrição:** Idade do paciente (valor numérico).
* **Relevância:** É um dos fatores mais importantes — risco de AVC aumenta drasticamente com a idade.

---

### **4. hypertension**

* **Descrição:** Indica se o paciente tem hipertensão.
* **Valores:**

  * **0** = não hipertenso
  * **1** = hipertenso
* **Relevância:** Hipertensão é um dos maiores fatores de risco para AVC.

---

### **5. heart_disease**

* **Descrição:** Indica presença de doenças cardíacas.
* **Valores:**

  * **0** = sem doenças
  * **1** = possui doença cardíaca
* **Relevância:** Altamente relevante, pois doenças cardiovasculares estão diretamente associadas ao risco de AVC.

---

### **6. ever_married**

* **Descrição:** Indica se a pessoa já foi casada.
* **Valores:** `"Yes"` ou `"No"`
* **Relevância:** Baixa. Geralmente não possui relação direta com AVC e pode ser descartada no modelo.

---

### **7. work_type**

* **Descrição:** Tipo de ocupação do paciente.
* **Valores possíveis:**

  * `"Private"`
  * `"Self-employed"`
  * `"Govt_job"`
  * `"Children"`
  * `"Never_worked"`
* **Relevância:** Pode refletir estilo de vida e rotina — moderadamente relevante.

---

### **8. Residence_type**

* **Descrição:** Local de residência.
* **Valores:** `"Urban"` ou `"Rural"`
* **Relevância:** Pode indicar acesso a serviços de saúde e perfil de risco ambiental.

---

### **9. avg_glucose_level**

* **Descrição:** Nível médio de glicose no sangue.
* **Relevância:** Altamente relevante — valores elevados indicam risco de diabetes, que aumenta chances de AVC.

---

### **10. bmi**

* **Descrição:** Índice de Massa Corporal (Body Mass Index).
* **Relevância:** Representa obesidade, sedentarismo e estado metabólico — fatores relevantes para AVC.

---

### **11. smoking_status**

* **Descrição:** Situação tabagística do paciente.
* **Valores:**

  * `"formerly smoked"`
  * `"never smoked"`
  * `"smokes"`
  * `"Unknown"`
* **Relevância:** Extremamente importante. O tabagismo é um forte fator de risco.

---

### **12. stroke**

* **Descrição:** Indica se o paciente já sofreu AVC.
* **Valores:**

  * **0** = não teve AVC
  * **1** = teve AVC
* **Uso:** **É a variável alvo (target)** do modelo de IA.

---

## 🧪 Pipeline do Projeto

1. **Limpeza dos dados**

   * Remover/ajustar valores faltantes
   * Tratar categorias
   * Remover outliers (ex.: BMI)
   * Analisar correlações

2. **Treinamento do modelo**

   * Teste com diferentes algoritmos (RandomForest, XGBoost, Logistic Regression etc.)

3. **Validação**

   * AUC-ROC
   * Matriz de confusão
   * Precision e Recall
   * Feature Importance

4. **Predição**

   * Dado um paciente → modelo retorna probabilidade (%) de AVC.

---

## 📦 Tecnologias usadas

* Python
* Pandas
* Scikit-Learn
* Matplotlib / Seaborn
* Jupyter Notebook
* (Opcional) Flask / FastAPI para API de predição
* (Opcional) Streamlit para interface web

---

## 📈 Resultado Esperado

Ao final, o modelo será capaz de:

* Receber dados clínicos e demográficos
* Processar automaticamente
* Retornar:
  → **Probabilidade de AVC (%)**
  → **Variáveis mais importantes para o risco individual**

---

## 📬 Como Executar

```bash
pip install -r requirements.txt
jupyter notebook
```

