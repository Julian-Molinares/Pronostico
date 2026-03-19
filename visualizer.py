import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Crea visualizaciones para análisis de pronósticos"""
    
    @staticmethod
    def plot_training_history(history, target_name="Modelo"):
        """
        Visualiza el historial de entrenamiento
        
        Args:
            history: Objeto history de Keras
            target_name: Nombre del objetivo para el título
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pérdida
        axes[0].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validación', linewidth=2)
        axes[0].set_title(f'Pérdida (MSE) - {target_name}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Épocas')
        axes[0].set_ylabel('MSE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(history.history['mae'], label='Entrenamiento', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validación', linewidth=2)
        axes[1].set_title(f'Error Absoluto Medio (MAE) - {target_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Épocas')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{target_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"✓ Gráfico de historial guardado: training_history_{target_name}.png")
    
    @staticmethod
    def plot_predictions_vs_actual(y_true, y_pred, set_name="Test", target_name="y1"):
        """
        Compara predicciones vs valores reales
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            set_name: Nombre del conjunto (Train/Test/Val)
            target_name: Nombre de la variable objetivo
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de línea
        axes[0].plot(y_true, label='Real', alpha=0.7, linewidth=2, marker='o', markersize=3)
        axes[0].plot(y_pred, label='Predicción', alpha=0.7, linewidth=2, marker='s', markersize=3)
        axes[0].set_title(f'Pronóstico vs Real - {target_name} ({set_name})', 
                        fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Muestras')
        axes[0].set_ylabel('Valor')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico de dispersión
        axes[1].scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Línea de referencia perfecta
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
        
        axes[1].set_xlabel('Valores Reales')
        axes[1].set_ylabel('Predicciones')
        axes[1].set_title(f'Dispersión: Real vs Predicción - {target_name}', 
                        fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(f'predictions_{target_name}_{set_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f" Gráfico de predicciones guardado: predictions_{target_name}_{set_name}.png")
    
    @staticmethod
    def plot_error_distribution(y_true, y_pred, target_name="y1"):
        """
        Visualiza la distribución de errores
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            target_name: Nombre de la variable objetivo
        """
        errors = y_true.flatten() - y_pred.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma
        axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
        axes[0].axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, 
                    label=f'Media = {np.mean(errors):.4f}')
        axes[0].set_xlabel('Error (Real - Predicción)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title(f'Distribución de Errores - {target_name}', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1])
        axes[1].set_title(f'Q-Q Plot - {target_name}', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'error_distribution_{target_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f" Gráfico de errores guardado: error_distribution_{target_name}.png")
    
    @staticmethod
    def plot_dual_targets(y1_true, y1_pred, y2_true, y2_pred, set_name="Test"):
        """
        Visualiza predicciones para ambos objetivos (y1 y y2)
        
        Args:
            y1_true, y1_pred: Valores reales y predicciones para y1
            y2_true, y2_pred: Valores reales y predicciones para y2
            set_name: Nombre del conjunto
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # y1 - Línea
        axes[0, 0].plot(y1_true, label='Real', alpha=0.7, linewidth=2)
        axes[0, 0].plot(y1_pred, label='Predicción', alpha=0.7, linewidth=2)
        axes[0, 0].set_title(f' y1 Heating Load - Línea ({set_name})', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Carga Térmica')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # y1 - Dispersión
        axes[0, 1].scatter(y1_true, y1_pred, alpha=0.6, s=50, color='orangered')
        min_val = min(y1_true.min(), y1_pred.min())
        max_val = max(y1_true.max(), y1_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 1].set_xlabel('Real')
        axes[0, 1].set_ylabel('Predicción')
        axes[0, 1].set_title(f' y1 Heating Load - Dispersión', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # y2 - Línea
        axes[1, 0].plot(y2_true, label='Real', alpha=0.7, linewidth=2, color='cyan')
        axes[1, 0].plot(y2_pred, label='Predicción', alpha=0.7, linewidth=2, color='blue')
        axes[1, 0].set_title(f' y2 Cooling Load - Línea ({set_name})', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Muestras')
        axes[1, 0].set_ylabel('Carga de Enfriamiento')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # y2 - Dispersión
        axes[1, 1].scatter(y2_true, y2_pred, alpha=0.6, s=50, color='cyan')
        min_val = min(y2_true.min(), y2_pred.min())
        max_val = max(y2_true.max(), y2_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 1].set_xlabel('Real')
        axes[1, 1].set_ylabel('Predicción')
        axes[1, 1].set_title(f' y2 Cooling Load - Dispersión', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'dual_targets_{set_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f" Gráfico dual guardado: dual_targets_{set_name}.png")