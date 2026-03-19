import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Carga y prepara datos de Energy Efficiency para pronóstico con LSTM
    
    Dataset: 768 muestras
    Features: X1-X8 (8 características)
    Targets: Y1 (Heating Load), Y2 (Cooling Load)
    """
    
    # ✅ CORREGIDO: Columnas en MAYÚSCULAS según el dataset real
    FEATURE_COLUMNS = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    TARGET_COLUMNS = ['Y1', 'Y2']  # ← Cambio de minúsculas a mayúsculas
    
    def __init__(self, filepath):
        """
        Inicializa el DataLoader
        
        Args:
            filepath (str): Ruta del archivo CSV de energy efficiency
        """
        self.filepath = filepath
        self.data = None
        self.original_data = None
        self.scalers = {}  # Para mantener escaladores separados por variable
        logger.info(f"DataLoader inicializado para: {filepath}")
    
    def load_data(self):
        """
        Carga datos desde archivo CSV
        
        Returns:
            pd.DataFrame: Datos cargados
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.original_data = self.data.copy()
            
            logger.info(f"✓ Datos cargados exitosamente")
            logger.info(f"  Dimensiones: {self.data.shape[0]} muestras × {self.data.shape[1]} columnas")
            logger.info(f"  Características: {', '.join(self.FEATURE_COLUMNS)}")
            logger.info(f"  Objetivos: {', '.join(self.TARGET_COLUMNS)}")
            logger.info(f"  Estadísticas básicas:\n{self.data.describe()}")
            
            return self.data
        except FileNotFoundError:
            logger.error(f"❌ Archivo no encontrado: {self.filepath}")
            raise
        except Exception as e:
            logger.error(f"❌ Error al cargar datos: {e}")
            raise
    
    def get_data_info(self):
        """Retorna información detallada del dataset"""
        if self.data is None:
            logger.warning("Los datos aún no han sido cargados")
            return None
        
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistics': self.data.describe().to_dict()
        }
        return info
    
    def normalize_data(self, variables=None, separate_scalers=True):
        """
        Normaliza los datos entre 0 y 1
        
        Args:
            variables (list): Variables a normalizar. Si es None, usa todas.
            separate_scalers (bool): Si True, usa escaladores separados por variable
        
        Returns:
            np.ndarray: Datos normalizados
        """
        if self.data is None:
            logger.error("Primero debe cargar los datos con load_data()")
            return None
        
        if variables is None:
            variables = self.FEATURE_COLUMNS + self.TARGET_COLUMNS
        
        data_to_normalize = self.data[variables].copy()
        
        if separate_scalers:
            # Crear escalador separado para cada variable
            normalized_arrays = []
            for col in variables:
                scaler = MinMaxScaler()
                scaled_col = scaler.fit_transform(data_to_normalize[[col]])
                self.scalers[col] = scaler
                normalized_arrays.append(scaled_col)
            
            normalized_data = np.hstack(normalized_arrays)
        else:
            # Un solo escalador para todos los datos
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data_to_normalize)
            self.scalers['general'] = scaler
        
        logger.info(f"✓ Datos normalizados: rango [0, 1]")
        logger.info(f"  Variables normalizadas: {variables}")
        
        return normalized_data
    
    def create_sequences(self, data, sequence_length, num_features, target_indices):
        """
        Crea secuencias de tiempo para LSTM
        
        Args:
            data (np.ndarray): Array de datos normalizados
            sequence_length (int): Longitud de la secuencia temporal
            num_features (int): Número de características
            target_indices (list/int): Índices de la(s) columna(s) objetivo
        
        Returns:
            tuple: (X, y) donde X es entrada y y es salida
        """
        X, y = [], []
        
        # Convertir target_indices a lista si es un solo número
        if isinstance(target_indices, int):
            target_indices = [target_indices]
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length, :num_features])
            y.append(data[i+sequence_length, target_indices])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✓ Secuencias creadas:")
        logger.info(f"  Total de muestras: {len(X)}")
        logger.info(f"  Forma de X: {X.shape} (muestras, pasos de tiempo, características)")
        logger.info(f"  Forma de y: {y.shape} (muestras, número de objetivos)")
        
        return X, y
    
    def split_data(self, X, y, train_ratio=0.8):
        """
        Divide datos en conjunto de entrenamiento y prueba
        
        Args:
            X (np.ndarray): Características
            y (np.ndarray): Objetivos
            train_ratio (float): Proporción de entrenamiento (0-1)
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"✓ División de datos:")
        logger.info(f"  Entrenamiento: {len(X_train)} muestras ({train_ratio*100:.0f}%)")
        logger.info(f"  Prueba: {len(X_test)} muestras ({(1-train_ratio)*100:.0f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, normalized_data, columns):
        """
        Invierte la normalización para obtener valores originales
        
        Args:
            normalized_data (np.ndarray): Datos normalizados
            columns (list): Nombres de las columnas
        
        Returns:
            np.ndarray: Datos en escala original
        """
        try:
            if len(columns) == 1 and columns[0] in self.scalers:
                scaler = self.scalers[columns[0]]
                # Crear matriz con dimensiones correctas
                temp = np.zeros((normalized_data.shape[0], 1))
                temp[:, 0] = normalized_data.flatten()
                return scaler.inverse_transform(temp)
            elif 'general' in self.scalers:
                return self.scalers['general'].inverse_transform(normalized_data)
            else:
                logger.warning("No se encontraron escaladores para invertir transformación")
                return normalized_data
        except Exception as e:
            logger.error(f"Error al invertir transformación: {e}")
            return normalized_data