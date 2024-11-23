# Proyecto de Clasificación de Vinos

## Contenido
- [Introducción](#introducción)
- [Dataset](#dataset)
- [Objetivo](#objetivo)
- [Enfoque](#enfoque)
- [Características y Metodología](#características-y-metodología)
- [Métricas y Resultados](#métricas-y-resultados)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Agradecimientos](#agradecimientos)

---

## Introducción
Este proyecto implementa una solución de machine learning para la clasificación de variedades de vino basándose en sus atributos químicos. Se desarrolló utilizando **Python** y **Databricks**, aprovechando técnicas avanzadas de preprocesamiento y pipelines de aprendizaje automático para ofrecer predicciones precisas.

El objetivo es demostrar un enfoque profesional en el manejo de datos, entrenamiento de modelos y la integración con **MLflow** para experimentación y trazabilidad.

---

## Dataset
El dataset utilizado es el **Wine Dataset** del UCI Machine Learning Repository. Contiene 178 muestras de vino, cada una descrita por 13 atributos químicos y una etiqueta objetivo que representa la variedad de vino (1, 2 o 3).

### Atributos
- Características químicas como Alcohol, Ácido Málico, Ceniza, etc.
- Etiquetas objetivo (1, 2, 3) que representan las variedades de vino.

El dataset se obtuvo desde:  
[Wine Dataset - UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)

---

## Objetivo
Desarrollar un pipeline robusto de machine learning que:
1. Preprocese el dataset, manejando desbalances de clases y valores atípicos.
2. Entrene múltiples modelos de clasificación para predecir la variedad de vino.
3. Evalúe los modelos con métricas como accuracy, precisión, recall y F1-score.
4. Registre experimentos y artefactos utilizando **MLflow** para trazabilidad y gestión de modelos.

---

## Enfoque
El flujo de trabajo incluye:
1. **Carga y Análisis de Datos**:
   - Visualización de la distribución de clases.
   - Identificación y tratamiento de valores atípicos usando Isolation Forest.
   - Análisis de correlación entre características.
2. **Preprocesamiento de Datos**:
   - Balanceo de clases mediante SMOTE.
   - Escalado de características numéricas.
3. **Entrenamiento y Evaluación**:
   - Modelos probados: Regresión Logística, Random Forest, Support Vector Machine (SVM), entre otros.
   - Validación cruzada y ajuste de hiperparámetros.
4. **Integración con MLflow**:
   - Registro de métricas y artefactos de los modelos.
   - Registro del mejor modelo.
5. **Visualización**:
   - Curvas de aprendizaje, matrices de confusión y curvas ROC.

---

## Características y Metodología
### Preprocesamiento
- **Detección de Valores Atípicos**: Isolation Forest detectó y eliminó valores anómalos.
- **Balanceo de Clases**: SMOTE equilibró las distribuciones de clases.

### Entrenamiento de Modelos
- **Algoritmos**:
  - Regresión Logística
  - Random Forest
  - Support Vector Machine (SVM)
  - Otros modelos base
- **Métricas de Evaluación**:
  - Accuracy
  - Precisión
  - Recall
  - F1-Score

### MLflow
- Registro de experimentos y métricas.
- Guardado de artefactos (ej. matrices de confusión, mapas de calor).
- Registro del modelo con mejor desempeño para futuras implementaciones.

---

## Métricas y Resultados
Los resultados detallados, incluidas métricas y visualizaciones, están documentados en el archivo **PDF adjunto**.

El modelo con mejor desempeño logró:
- **Accuracy**: 98%
- **F1-Score**: 97%

Todos los artefactos, como curvas de aprendizaje y matrices de confusión, fueron registrados en MLflow.

---

## Cómo Ejecutar
### Requisitos
- **Databricks Community Edition**
- Python 3.8+
- Librerías:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`
  - `imbalanced-learn`
  - `mlflow`

### Pasos
1. Clona el repositorio:
   ```bash
   git clone https://github.com/tuusuario/Wine_Classification.git
2. Carga el archivo Wine_Classification.ipynb en Databricks.
3. Ejecuta el notebook para reproducir los resultados.
4. Revisa los experimentos registrados en MLflow.

