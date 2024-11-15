import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('task-1-stocks.csv')  # Имя файла с данными
returns = data.pct_change().dropna()  # Ежедневная доходность

# Средняя доходность и ковариация доходностей акций
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Уровень риска и начальные веса
risk_tolerance = 0.2
num_assets = len(mean_returns)
initial_weights = np.ones(num_assets) / num_assets  # Равномерное распределение

# Ограничения для весов
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))

# Функции для оптимизации
def portfolio_return(weights):
    return np.dot(weights, mean_returns) * 100  # Доходность в %

def portfolio_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Стандартное отклонение (риск)

def objective_function(weights):
    return -portfolio_return(weights)  # Минус для максимизации доходности

# Оптимизация
result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Получение оптимальных весов и расчёт доходности и риска
optimal_weights = result.x
optimal_return = portfolio_return(optimal_weights)
optimal_risk = portfolio_risk(optimal_weights)

# Вывод результатов
allocation = pd.DataFrame({'Акция': data.columns, 'Процент вложений': optimal_weights * 100})
print("Распределение по акциям:")
print(allocation)

print("\nДоходность портфеля за 100 дней:", optimal_return)
print("Риск портфеля:", optimal_risk)

# Построение графиков
# 1. Улучшенный график доходности и риска для каждой акции
plt.figure(figsize=(12, 7))
plt.scatter(cov_matrix.values.diagonal() ** 0.5, mean_returns * 100, marker='o', s=100, color='skyblue')
for i, txt in enumerate(data.columns):
    plt.annotate(txt,
                 (cov_matrix.values[i, i] ** 0.5, mean_returns[i] * 100),
                 textcoords="offset points",
                 xytext=(5, 5),  # Смещение подписей
                 ha='center',
                 arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

plt.xlabel("Риск (Стандартное отклонение)")
plt.ylabel("Доходность (%)")
plt.title("Доходность и риск для каждой акции")
plt.show()

# 2. График зависимости доходности портфеля от уровня допустимого риска
target_risks = np.linspace(0.05, 0.3, 50)
target_returns = []

for risk in target_risks:
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'ineq', 'fun': lambda weights: risk - portfolio_risk(weights)}]
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    target_returns.append(-result.fun)

plt.figure(figsize=(10, 6))
plt.plot(target_risks, target_returns, '-o', color='b')
plt.xlabel("Допустимый риск (стандартное отклонение)")
plt.ylabel("Доходность (%)")
plt.title("Зависимость доходности портфеля от уровня допустимого риска")
plt.show()
