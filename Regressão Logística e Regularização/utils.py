import sys
import numpy as np
from matplotlib import pyplot

sys.path.append('..')

def mapFeature(X1, X2, degree=6):
    """
    Mapeia dois features de entrada em features quadráticos utilizados
    no exercício de regularização.

    Retorna um novo feature array com mais features, contemplando
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parâmetros
    ----------
    X1 : tipo_array
        Um vetor na forma (m, 1), contendo um feature para todas as amostras.

    X2 : tipo_array
        Um vetor na forma (m, 1), contendo um feature para todas as amostras.
        As entradas X1, X2 devem possuir o mesmo tamanho.

    degree: int, optional
        Ordem poliniomial.

    Returns
    -------
    : tipo array
        Uma matriz com m linhas, e número de colunas dependendo do grau polinomial
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


def plotDecisionBoundary(plotData, theta, X, y):
    """
    Imprime os pontos de dado X e y em uma nova figura com a linha divisória de decisão definida
    por theta.

    Parâmetros
    ----------
    plotData : func
        Função referência para imprimir os dados X e y.

    theta : tipo_array
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : tipo_array
        Conjunto de entrada. assume-se que X deve ser uma das abaixo:
            1) matriz (m, 3), onde a primeira coluna é uma coluna de 1's.
            2) matriz (m, n), n>3, onde a primeira coluna é uma coluna de 1's.

    y : array_like
        Vetor com o rótulo de forma (m, ).
    """
    # Garanta que theta é um tipo array
    theta = np.array(theta)

    # Imprime conjunto de dados (lembre-se que o conjunto de dados são termos de interceptação)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # escolhe dois pontos para definir a linha divisória
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # calcula linha divisória de decisão
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Imprime e ajusta eixos para melhor visualização
        pyplot.plot(plot_x, plot_y)

        # Legendas específicas para essa tarefa
        pyplot.legend(['Aceito', 'Não aceito', 'Contorno de decisão'])
        pyplot.xlim([30, 100])
        pyplot.ylim([30, 100])
    else:
        # Limites para o grid
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Calcula z = theta*x sobre o grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # Importante transpor z antes de imprimir o contorno
        # print(z)

        # Imprime z = 0
        pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='darkgrey')
        pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greys', alpha=0.4)
