import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, input_shape, output_dim=1, lstm_units=50, 
                dropout_rate=0.2, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        logger.info(f"Inicializando LSTMModel:")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Output dim: {output_dim}")
        logger.info(f"LSTM units: {lstm_units}")
        logger.info(f"Dropout rate: {dropout_rate}")
        
        self._build_model()
    
    def _build_model(self):
        self.model = Sequential([
            LSTM(self.lstm_units, activation='relu', 
                input_shape=self.input_shape, return_sequences=True,
                name='lstm_1'),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units, activation='relu', 
                return_sequences=False, name='lstm_2'),
            Dropout(self.dropout_rate),
            
            Dense(25, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate),
            
            Dense(self.output_dim, activation='linear', name='output')
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info("Modelo LSTM construido exitosamente")
        self._log_model_summary()
    
    def _log_model_summary(self):
        self.model.summary(print_fn=logger.info)
    
    def train(self, X_train, y_train, X_val, y_val, 
            epochs=100, batch_size=16, patience=15):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        logger.info(f"Iniciando entrenamiento...")
        logger.info(f"Épocas: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Paciencia EarlyStopping: {patience}")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Entrenamiento completado")
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=0)
        loss = results[0]
        mae = results[1]
        mse = results[2]
        
        logger.info(f"Evaluación en conjunto de prueba:")
        logger.info(f"Pérdida (MSE): {loss:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        
        return loss, mae, mse
    
    def get_model(self):
        return self.model
    
    def save_model(self, filepath):
        self.model.save(filepath)
        logger.info(f" Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f" Modelo cargado desde: {filepath}")