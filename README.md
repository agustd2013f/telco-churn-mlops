<<<<<<< HEAD
# Proyecto Telco Churn - Modelo Principal: Random Forest

=======
# Proyecto Telco Churn - Modelo Principal: SVM
>>>>>>> test/conflicto
Proyecto base para la clasificación de fuga de clientes (Churn) utilizando algoritmos en Python.

## Estructura del Proyecto
- `data/telco_demo.csv`: Muestra sintética del dataset.
- `src/train.py`: Script de entrenamiento de modelos.
- `requirements.txt`: Librerías requeridas.
## Evaluación de Métricas
Se evalúa la precisión con Accuracy, F1-Score y ROC-AUC.

## Modelo SVM
Se incluye un modelo Support Vector Machine (RBF) estandarizado con StandardScaler.

## Métricas de Clasificación
- **Accuracy:** Proporción total de predicciones correctas.
- **Precision:** Exactitud de las predicciones positivas de abandono (Churn).
- **Recall:** Capacidad del modelo para detectar todos los casos reales de Churn.
- **F1-Score:** Media armónica entre Precision y Recall. 
## Resultados del Proyecto 
El modelo SVM con kernel RBF obtuvo el mejor desempeno en ROC-AUC y F1-Score. 
