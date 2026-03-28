import os
import logging
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
from multi_target_forecaster import MultiTargetForecaster
from visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('energy_forecasting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    FILEPATH = 'energy_efficiency_data.csv'
    SEQUENCE_LENGTH = 10
    PREDICT_BOTH = True
    
    # Hyperparámetros del modelo
    LSTM_UNITS = 64
    EPOCHS = 100
    BATCH_SIZE = 16
    
    try:
        forecaster = MultiTargetForecaster(
            filepath=FILEPATH,
            sequence_length=SEQUENCE_LENGTH,
            predict_both=PREDICT_BOTH
        )
        
        # Ejecutar pipeline
        metrics, y_train_pred, y_test_pred = forecaster.run_complete_pipeline(
            lstm_units=LSTM_UNITS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # Visualizaciones
        logger.info("\nGenerando visualizaciones...")
        
        # Historial de entrenamiento
        if forecaster.model.history:
            Visualizer.plot_training_history(
                forecaster.model.history,
                target_name="Energy Efficiency"
            )
        
        # Obtener datos de prueba
        X_test = forecaster.test_data['X']
        y_test = forecaster.test_data['y']
        X_train = forecaster.training_data['X']
        y_train = forecaster.training_data['y']
        
        if PREDICT_BOTH:
            # Predicciones para Y1 (Heating Load)
            Visualizer.plot_predictions_vs_actual(
                y_test[:, 0], y_test_pred[:, 0],
                set_name="Prueba", target_name="Y1 - Heating Load"
            )
            
            Visualizer.plot_error_distribution(
                y_test[:, 0], y_test_pred[:, 0],
                target_name="Y1 - Heating Load"
            )
            
            # Predicciones para Y2 (Cooling Load)
            Visualizer.plot_predictions_vs_actual(
                y_test[:, 1], y_test_pred[:, 1],
                set_name="Prueba", target_name="Y2 - Cooling Load"
            )
            
            Visualizer.plot_error_distribution(
                y_test[:, 1], y_test_pred[:, 1],
                target_name="Y2 - Cooling Load"
            )
            
            # Gráfico dual
            Visualizer.plot_dual_targets(
                y_test[:, 0], y_test_pred[:, 0],
                y_test[:, 1], y_test_pred[:, 1],
                set_name="Prueba"
            )
        else:
            # Solo Y1
            Visualizer.plot_predictions_vs_actual(
                y_test, y_test_pred,
                set_name="Prueba", target_name="Y1 - Heating Load"
            )
            
            Visualizer.plot_error_distribution(
                y_test, y_test_pred,
                target_name="Y1 - Heating Load"
            )
        
        logger.info(" PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("\nArchivos generados:")
        logger.info("  - energy_forecasting.log")
        logger.info("  - training_history_*.png")
        logger.info("  - predictions_*.png")
        logger.info("  - error_distribution_*.png")
        logger.info("  - dual_targets_*.png (si predict_both=True)")
        
    except Exception as e:
        logger.error(f" Error en el proceso: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()