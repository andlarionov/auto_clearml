import pandas as pd

# Открываем файл
df = pd.read_csv('data/shopping_trends.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

df = df[['age', 'gender', 'category', 'size', 'subscription_status', 'discount_applied']]

# Сохраняем данные в формате CSV
df.to_csv('data/df.csv', index=False)