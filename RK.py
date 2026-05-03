import sklearn as skl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение датасета и преобразование его в тип DataFrame
data = skl.datasets.load_iris()
y = data['target']
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()

# Добавляем target в DataFrame
df_with_target = df.copy()
df_with_target['species'] = y

# Строим матрицу корреляции
plt.figure(figsize=(8, 6))
sns.heatmap(df_with_target.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Корреляционная матрица (признаки + target)')
plt.show()

# Группируем по видам и считаем средние значения
df_with_target = df.copy()
df_with_target['species'] = y

# Средние значения признаков для каждого вида
mean_by_species = df_with_target.groupby('species').mean()

# Строим гистограмму
mean_by_species.T.plot(kind='bar', figsize=(10, 6))
plt.title('Средние значения признаков по видам ирисов')
plt.xlabel('Признаки')
plt.ylabel('Среднее значение (см)')
plt.legend(title='Вид', labels=['Setosa', 'Versicolor', 'Virginica'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()