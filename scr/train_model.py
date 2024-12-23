import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
import matplotlib.pyplot as plt
from clearml import Task

# Инициализация ClearML Task
task = Task.init(project_name="Discount Prediction", task_name="Model Training and Comparison")

# Создаем папки для артефактов и отчетов, если они не существуют
artifacts_dir = 'artifacts'
reports_dir = 'reports'

# Открываем файл
df = pd.read_csv('data/df.csv')

# Разделяем признаки и целевую переменную
X = df.drop('discount_applied', axis=1)
y = df['discount_applied']

# Преобразуем целевую переменную с помощью LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделяем данные на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Преобразуем возрастные данные с помощью StandardScaler
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])

# Кодируем категориальные признаки с помощью OrdinalEncoder
ordinal = OrdinalEncoder()
X_train[['gender', 'category', 'size', 'subscription_status']] = ordinal.fit_transform(
    X_train[['gender', 'category', 'size', 'subscription_status']]
)
X_test[['gender', 'category', 'size', 'subscription_status']] = ordinal.transform(
    X_test[['gender', 'category', 'size', 'subscription_status']]
)

# Параметры первой модели
model_params_1 = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
}

# Логирование параметров первой модели в ClearML
task.connect(model_params_1)

# Обучаем первую модель
model_1 = RandomForestClassifier(**model_params_1)
model_1.fit(X_train, y_train)

# Предсказание и метрика для первой модели
y_pred_1 = model_1.predict(X_test)
f1_1 = f1_score(y_test, y_pred_1)
task.get_logger().report_scalar("F1 Score", "Model 1", iteration=1, value=f1_1)

# Сохранение первой модели и логирование артефакта
model_1_filepath = os.path.join(artifacts_dir, 'model_1.pkl')
joblib.dump(model_1, model_1_filepath)
task.upload_artifact(name="model_1.pkl", artifact_object=model_1_filepath)

# Параметры второй модели
model_params_2 = {
    'n_estimators': 250,
    'max_depth': None,
    'min_samples_split': 15,
    'min_samples_leaf': 7,
}

# Логирование параметров второй модели в ClearML
task.connect(model_params_2)

# Обучаем вторую модель
model_2 = RandomForestClassifier(**model_params_2)
model_2.fit(X_train, y_train)

# Предсказание и метрика для второй модели
y_pred_2 = model_2.predict(X_test)
f1_2 = f1_score(y_test, y_pred_2)
task.get_logger().report_scalar("F1 Score", "Model 2", iteration=2, value=f1_2)

# Сохранение второй модели и логирование артефакта
model_2_filepath = os.path.join(artifacts_dir, 'model_2.pkl')
joblib.dump(model_2, model_2_filepath)
task.upload_artifact(name="model_2.pkl", artifact_object=model_2_filepath)

# Параметры третьей модели
model_params_3 = {
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 250,
    'C': 1.0
}

# Логирование параметров третьей модели в ClearML
task.connect(model_params_3)

# Обучаем третью модель
model_3 = LogisticRegression(**model_params_3)
model_3.fit(X_train, y_train)

# Предсказание и метрика для третьей модели
y_pred_3 = model_3.predict(X_test)
f1_3 = f1_score(y_test, y_pred_3)
task.get_logger().report_scalar("F1 Score", "Model 3", iteration=3, value=f1_3)

# Сохранение третьей модели и логирование артефакта
model_3_filepath = os.path.join(artifacts_dir, 'model_3.pkl')
joblib.dump(model_3, model_3_filepath)
task.upload_artifact(name="model_3.pkl", artifact_object=model_3_filepath)

# Данные для графика
models = ['Model 1', 'Model 2', 'Model 3']
f1_scores = [f1_1, f1_2, f1_3]

# Построение и логирование графика
plt.bar(models, f1_scores)
plt.ylabel('F1 Score')
plt.title('Comparison of Model Performance')

# Сохранение графика в папке reports
plot_filepath = os.path.join(reports_dir, 'performance_comparison.png')
plt.savefig(plot_filepath)

# Логируем график как артефакт в ClearML
task.upload_artifact(name="performance_comparison.png", artifact_object=plot_filepath)

# Логируем текстовый отчет о производительности моделей в папку reports
performance_report = f"F1-score Model 1: {f1_1}\nF1-score Model 2: {f1_2}\nF1-score Model 3: {f1_3}"
report_filepath = os.path.join(reports_dir, 'performance_report.txt')

# Сохранение текста отчета
with open(report_filepath, 'w') as report_file:
    report_file.write(performance_report)

# Логируем текстовый отчет в ClearML
task.get_logger().report_text(performance_report)

# Завершение Task
task.close()
