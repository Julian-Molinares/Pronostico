import numpy as np
import logging
from data_loader import DataLoader
from lstm_model import LSTMModel
from evaluator import Evaluator

logger = logging.getLogger(__name__)

class MultiTargetForecaster:
    def __init__(self, filepath, sequence_length=10, predict_both=True):
        self.filepath = filepath
        self.sequence_length = sequence_length
        self.predict_both = predict_both
        
        self.data_loader = DataLoader(filepath)
        self.models = {}
        self.training_data = {}
        self.test_data = {}
        
        logger.info(" MultiTargetForecaster inicializado")
        logger.info(f"  Archivo: {filepath}")
        logger.info(f"  Secuencia temporal: {sequence_length}")
        logger.info(f"  Predicción multitarget: {predict_both}")
    
    def prepare_data(self):
        # Cargar datos
        self.data_loader.load_data()
        
        # Normalizar
        normalized_data = self.data_loader.normalize_data(
            variables=DataLoader.FEATURE_COLUMNS + DataLoader.TARGET_COLUMNS,
            separate_scalers=True
        )
        
        num_features = len(DataLoader.FEATURE_COLUMNS)
        
        if self.predict_both:
            # Predecir Y1 (índice 8) e Y2 (índice 9)
            target_indices = [8, 9]
            output_dim = 2
            logger.info("Objetivo: Predicción dual (Y1 - Heating Load, Y2 - Cooling Load)")
        else:
            # Solo predecir Y1 (índice 8)
            target_indices = 8
            output_dim = 1
            logger.info("Objetivo: Predicción simple (Y1 - Heating Load)")
        
        # Crear secuencias
        X, y = self.data_loader.create_sequences(
            normalized_data,
            self.sequence_length,
            num_features,
            target_indices
        )
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.data_loader.split_data(
            X, y, train_ratio=0.8
        )
        
        # Guardar en diccionario
        self.training_data = {
            'X': X_train,
            'y': y_train,
            'normalized': normalized_data
        }
        self.test_data = {
            'X': X_test,
            'y': y_test
        }
        
        logger.info("Preparación de datos completada")
        
        return X_train, X_test, y_train, y_test, output_dim
    
    def build_and_train_model(self, lstm_units=64, epochs=100, batch_size=16):
        X_train = self.training_data['X']
        y_train = self.training_data['y']
        X_test = self.test_data['X']
        y_test = self.test_data['y']
        
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Crear modelo
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = LSTMModel(
            input_shape=input_shape,
            output_dim=output_dim,
            lstm_units=lstm_units,
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Entrenar
        self.model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            patience=15
        )
        
        logger.info("Modelo entrenado exitosamente")
    
    def evaluate_model(self):
        X_train = self.training_data['X']
        y_train = self.training_data['y']
        X_test = self.test_data['X']
        y_test = self.test_data['y']
        
        # Predicciones
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Evaluar
        if self.predict_both:
            # Separar predicciones para Y1 y Y2
            print("\nEVALUACIÓN - Y1 (HEATING LOAD)")
            
            metrics_y1_train = Evaluator.display_metrics(
                y_train[:, 0], y_train_pred[:, 0],
                "Entrenamiento", "- Y1 Heating Load"
            )
            metrics_y1_test = Evaluator.display_metrics(
                y_test[:, 0], y_test_pred[:, 0],
                "Prueba", "- Y1 Heating Load"
            )
            
            print("\nEVALUACIÓN - Y2 (COOLING LOAD)")
            
            metrics_y2_train = Evaluator.display_metrics(
                y_train[:, 1], y_train_pred[:, 1],
                "Entrenamiento", "- Y2 Cooling Load"
            )
            metrics_y2_test = Evaluator.display_metrics(
                y_test[:, 1], y_test_pred[:, 1],
                "Prueba", "- Y2 Cooling Load"
            )
            
            return {
                'Y1': {'train': metrics_y1_train, 'test': metrics_y1_test},
                'Y2': {'train': metrics_y2_train, 'test': metrics_y2_test}
            }, y_train_pred, y_test_pred
        else:
            metrics_train = Evaluator.display_metrics(
                y_train, y_train_pred,
                "Entrenamiento", "- Y1 Heating Load"
            )
            metrics_test = Evaluator.display_metrics(
                y_test, y_test_pred,
                "Prueba", "- Y1 Heating Load"
            )
            
            return {
                'Y1': {'train': metrics_train, 'test': metrics_test}
            }, y_train_pred, y_test_pred
    
    def run_complete_pipeline(self, lstm_units=64, epochs=100, batch_size=16):
        print("\nPRONÓSTICO LSTM - ENERGY EFFICIENCY DATASET")
        print(f"\nDataset: Energy Efficiency (768 muestras, 8 características)")
        print(f"Variables objetivo: Y1 (Heating Load), Y2 (Cooling Load)")
        print(f"Modelo: LSTM multitarget con regularización")
        
        # Fase 1: Preparación
        self.prepare_data()
        
        # Fase 2: Entrenamiento
        self.build_and_train_model(lstm_units=lstm_units, epochs=epochs, batch_size=batch_size)
        
        # Fase 3: Evaluación
        metrics, y_train_pred, y_test_pred = self.evaluate_model()
        
        return metrics, y_train_pred, y_test_pred