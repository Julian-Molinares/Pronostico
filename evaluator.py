import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, mean_absolute_percentage_error
)
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, metric_name="Predicción"):
        # Asegurar que son arrays 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calcular RMSE normalizado
        y_mean = np.mean(y_true)
        normalized_rmse = rmse / y_mean if y_mean != 0 else 0
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'NRMSE': normalized_rmse
        }
        
        logger.info(f"Métricas calculadas para {metric_name}")
        
        return metrics
    
    @staticmethod
    def display_metrics(y_true, y_pred, set_name="Resultados", target_name=""):
        metrics = Evaluator.calculate_metrics(y_true, y_pred, set_name)
        
        title = f"{'MÉTRICAS DE RENDIMIENTO':^60}"
        subtitle = f"{set_name} {target_name}".center(60)
        
        print(title)
        if target_name:
            print(subtitle)
        print(f"{'Métrica':<25} {'Valor':>20} {'Descripción':>15}")
        print(f"{'-'*60}")
        
        descriptions = {
            'MSE': 'Error cuadrático',
            'RMSE': 'Raíz del error',
            'MAE': 'Error absoluto',
            'MAPE': 'Error porcentual',
            'R²': 'Coef. determina.',
            'NRMSE': 'RMSE normalizado'
        }
        
        for metric, value in metrics.items():
            desc = descriptions.get(metric, '')
            print(f"{metric:<25} {value:>20.6f} {desc:>15}")
        
        return metrics
    
    @staticmethod
    def compare_metrics(results_dict):
        comparison = {}
        
        for name, (y_true, y_pred) in results_dict.items():
            metrics = Evaluator.calculate_metrics(y_true, y_pred, name)
            comparison[name] = metrics
            
            print(f"{name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")
            print()
        
        return comparison
    
    @staticmethod
    def get_error_statistics(y_true, y_pred):
        errors = y_true.flatten() - y_pred.flatten()
        
        stats = {
            'error_mean': np.mean(errors),
            'error_std': np.std(errors),
            'error_min': np.min(errors),
            'error_max': np.max(errors),
            'error_median': np.median(errors),
            'error_q25': np.percentile(errors, 25),
            'error_q75': np.percentile(errors, 75)
        }
        
        return stats