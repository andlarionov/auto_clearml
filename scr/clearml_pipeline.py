import os
from clearml.automation import PipelineController

# Проверка наличия ключей ClearML в переменных окружения
access_key = os.getenv("CLEARML_API_ACCESS_KEY")
secret_key = os.getenv("CLEARML_API_SECRET_KEY")

if not access_key or not secret_key:
    raise EnvironmentError("ClearML API keys are not set in the environment variables")

print(f"Access Key: {access_key}")
print(f"Secret Key: {secret_key}")

# Инициализация пайплайна
pipeline = PipelineController(
    name="Data Processing and Model Training Pipeline",
    project="Discount Prediction",
    version="1.0"
)

# Шаг 1: Обработка данных
pipeline.add_step(
    name="data_processing",
    base_task_project="Discount Prediction",
    base_task_name="Data Processing",
    parameter_override={
        "General/dataset_path": "data/shopping_trends.csv",
        "Output/output_path": "data/df.csv"
    }
)

# Шаг 2: Обучение моделей
pipeline.add_step(
    name="train_models",
    base_task_project="Discount Prediction",
    base_task_name="Train Models",
    parents=["data_processing"],
    parameter_override={
        "Input/data_path": "data/df.csv"
    }
)

# Запуск пайплайна
pipeline.start()
