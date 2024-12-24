from clearml import Task
from clearml.automation import PipelineController

# Инициализация задачи
task = Task.init(project_name='Discount Prediction', task_name='Pipeline Initialization', task_type='optimizer')

# Инициализация пайплайна
pipeline = PipelineController(
    name="Data Processing and Model Training Pipeline",
    project="Discount Prediction",
    version="1.0"
)

try:
    # Шаг 1: Обработка данных
    data_processing_task = pipeline.add_step(
        name="data_processing",
        base_task_project="Discount Prediction",
        base_task_name="Data Processing",
        parameter_override={
            "General/dataset_path": "data/shopping_trends.csv",
            "Output/output_path": "data/df.csv"
        }
    )
    
    if not data_processing_task:
        raise Exception("Failed to create data processing task")
    
    # Установка очереди для задачи обработки данных
    data_processing_task.set_base_task_queue('data_processing_queue')

    # Шаг 2: Обучение моделей
    train_models_task = pipeline.add_step(
        name="train_models",
        base_task_project="Discount Prediction",
        base_task_name="Model Training and Comparison",
        parents=["data_processing"],
        parameter_override={
            "Input/data_path": "data/df.csv"
        }
    )
    
    if not train_models_task:
        raise Exception("Failed to create training models task")
    
    # Установка очереди для задачи обучения моделей
    train_models_task.set_base_task_queue('model_training_queue')

except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Запуск пайплайна
    pipeline.start()

