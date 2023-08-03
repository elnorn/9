import numpy as np

def polyfit_least_squares(x, y, m):
    # Построение матрицы Вандермонда
    n = len(x)
    A = np.zeros((n, m+1))
    for i in range(n):
        for j in range(m+1):
            A[i,j] = x[i]**j

    # Вычисление решение по методу наименьших квадратов
    c = np.linalg.lstsq(A, y, rcond=None)[0]

        # Построение полиномиальной функции
    def poly(x):
        return sum(c[j]*x**j for j in range(m+1))

    return poly, c

# Пример
n = 10
m = 3
h = 1/n
x = np.linspace(0, 1, n+1)
y = 1 - np.cos(x)

poly, c = polyfit_least_squares(x, y, m)
print("Polynomial coefficients:", c)
print("Approximation error:", np.linalg.norm(poly(x) - y))