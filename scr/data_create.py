from clearml import Task
import pandas as pd

# Создание задачи для обработки данных
task = Task.init(project_name="Discount Prediction", task_name="Data Processing")

# Обработка данных
df = pd.read_csv('data/shopping_trends.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df[['age', 'gender', 'category', 'size', 'subscription_status', 'discount_applied']]
df.to_csv('data/df.csv', index=False)

task.close()