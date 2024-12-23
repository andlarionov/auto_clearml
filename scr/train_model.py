import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
import matplotlib.pyplot as plt
from clearml import Task, Logger

# Инициализация ClearML Task
task = Task.init(project_name="Discount Prediction", task_name="Model Training and Comparison")

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
Logger.current_logger().report_scalar("F1 Score", "Model 1", iteration=1, value=f1_1)

# Сохранение первой модели
joblib.dump(model_1, 'models/model_1.pkl')
Logger.current_logger().report_artifact("model_1.pkl")

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
Logger.current_logger().report_scalar("F1 Score", "Model 2", iteration=2, value=f1_2)

# Сохранение второй модели
joblib.dump(model_2, 'models/model_2.pkl')
Logger.current_logger().report_artifact("model_2.pkl")

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
Logger.current_logger().report_scalar("F1 Score", "Model 3", iteration=3, value=f1_3)

# Сохранение третьей модели
joblib.dump(model_3, 'models/model_3.pkl')
Logger.current_logger().report_artifact("model_3.pkl")

# Данные для графика
models = ['Model 1', 'Model 2', 'Model 3']
f1_scores = [f1_1, f1_2, f1_3]

plt.bar(models, f1_scores)
plt.ylabel('F1 Score')
plt.title('Comparison of Model Performance')
plt.savefig('performance_comparison.png')

# Логируем график в ClearML
Logger.current_logger().report_image(
    title="Model Performance Comparison", series="F1 Score", local_path='performance_comparison.png'
)

# Логируем текстовый отчет о производительности моделей
with open('model_performance_report.txt', 'w') as f:
    f.write(f"F1-score Model 1: {f1_1}\n")
    f.write(f"F1-score Model 2: {f1_2}\n")
    f.write(f"F1-score Model 3: {f1_3}\n")

Logger.current_logger().report_text("Model Performance Report", open('model_performance_report.txt').read())

print(f"F1-score Model 1: {f1_1}")
print(f"F1-score Model 2: {f1_2}")
print(f"F1-score Model 3: {f1_3}")

# Завершение Task
task.close()