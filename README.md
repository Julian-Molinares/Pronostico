# Predicción de Eficiencia Energética con LSTM

Este proyecto implementa un modelo de Inteligencia Artificial basado en redes neuronales recurrentes (LSTM) para predecir la **Carga Térmica (Heating Load - Y1)** y la **Carga de Enfriamiento (Cooling Load - Y2)** de diferentes diseños de edificios (Dataset de Eficiencia Energética).

## Características Principales

* **Modelo LSTM:** Arquitectura profunda con capas LSTM y Dropout para evitar sobreajuste.
* **Predicción Multi-Objetivo:** Capacidad de predecir una sola variable o ambas simultáneamente.
* **Procesamiento de Secuencias:** Transformación de datos tabulares a secuencias temporales para aprovechar dependencias.
* **Evaluación Completa:** Cálculo de métricas como MSE, RMSE, MAE, MAPE, R², NRMSE.
* **Visualizaciones Integradas:** Gráficos del historial de entrenamiento, comparativas Real vs Predicción, distribución de errores y Q-Q plots.
* **Logging:** Sistema de registro de eventos detallado (`energy_forecasting.log`).

## Estructura del Proyecto

* `main.py`: Punto de entrada del programa. Configura parámetros y orquesta la ejecución completa (Pipeline).
* `multi_target_forecaster.py`: Controlador principal que maneja la preparación de datos, entrenamiento y evaluación.
* `data_loader.py`: Carga el dataset, escala los datos (MinMaxScaler) y genera las secuencias temporales (X e y).
* `lstm_model.py`: Define, construye y entrena la topología de la red (Keras/TensorFlow) junto con callbacks (EarlyStopping, ReduceLROnPlateau).
* `evaluator.py`: Calcula y muestra por consola las métricas de rendimiento estadístico.
* `visualizer.py`: Genera y guarda todos los gráficos (PNG) del rendimiento del modelo y su precisión.

## Requisitos

Las dependencias principales del proyecto se encuentran en `requirements.txt`. Las bibliotecas más importantes son:
* `tensorflow` (o `keras`)
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `scipy`

Instalar las dependencias con:
```bash
pip install -r requirements.txt
```

## Uso

1. Asegúrate de tener el dataset `energy_efficiency_data.csv` en el directorio principal.
2. Ejecuta el archivo principal:
```bash
python main.py
```

Al finalizar, se habrán generado varios archivos `.png` con las gráficas de resultados, el archivo de logs `energy_forecasting.log` y el modelo guardado `best_model.h5`.
