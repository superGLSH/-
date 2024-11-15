from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from math import pi
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.optimize import minimize
import pandas as pd
from matplotlib import pyplot as plt
import pymorphy3
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

vectorizer = TfidfVectorizer(max_features=128)
banned = stopwords.words(
    "russian") + [",", ".", ";", ":", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
morph = pymorphy3.MorphAnalyzer()


def csv2vec(data):
    t_data = [word_tokenize(i) for i in data]  # tokenized data
    l_data = [" ".join(
        [morph.parse(t_data[i][j])[0].normal_form for j in range(len(t_data[i])) if t_data[i][j] not in banned]) for i
        in range(len(t_data))]  # lemmatized data
    matrix = vectorizer.fit_transform(pd.DataFrame(l_data)[0]).toarray()
    return matrix


data = pd.read_csv("task-3-dataset.csv")
reviews = data["отзывы"].to_list()
review_class = data["разметка"].to_list()


def func(x): return 1 if x == "+" else 0


review_class = np.array(list(map(func, review_class)))
matrix = csv2vec(reviews)

# переменные
n_q = 5
depth = 2
repetitions = 1000
teta_count = n_q * depth

def U_operator(circ, vector):
    for i in range(n_q):
        circ.rz(vector[i], i)

    for i in range(n_q - 1):
        circ.cx(i, i + 1)
        circ.rz((pi - vector[i]) * (pi - vector[i + 1]), i + 1)
        circ.cx(i, i + 1)
        circ.barrier()


def W_operator_x(circ, params):
    for layer in range(depth):
        if layer != 0:
            for k in range(n_q - 1):
                circ.cx(k, k + 1)

        for i in range(n_q):
            n = layer * n_q + i
            circ.rx(params[teta_count:][n], i)


def W_operator_y(circ, params):
    for layer in range(depth):
        if layer != 0:
            for k in range(n_q - 1):
                circ.cx(k, k + 1)

        for i in range(n_q):
            n = layer * n_q + i
            circ.ry(params[:teta_count][n], i)


def Anzaz(circ, vector, params):
    circ.h(range(n_q))
    U_operator(circ, vector)
    circ.h(range(n_q))
    U_operator(circ, vector)
    W_operator_y(circ, params)
    W_operator_x(circ, params)


def step(data, answer):
    def function(params):
        error = 0
        for Vector, ans in zip(data, answer):
            Qr = QuantumRegister(n_q, name="Q")
            Qc = QuantumCircuit(Qr)
            Anzaz(Qc, Vector , params)
            Qc.measure_all()

            """Qc.draw(output="mpl", initial_state=True)
            plt.show()"""

            simulator = AerSimulator()
            transpiled_qc = transpile(Qc, simulator)
            job = simulator.run(transpiled_qc, shots=repetitions)
            result = job.result()
            Counts = result.get_counts()

            cost = 0
            state_arr = list(Counts.keys())
            for state in state_arr:
                if state.count("1") % 2 == 0:
                    cost += Counts[state] / repetitions
                else:
                    cost -= Counts[state] / repetitions

            error += (ans - cost) ** 2
        err_arr.append(error)
        return np.sqrt(error)

    return function


for i in range(int(matrix.shape[1] / n_q)):
    err_arr = []
    D = matrix.T[i * n_q: (i + 1) * n_q]
    function = step(D, review_class[i * n_q: (i + 1) * n_q])
    Init_params = np.random.random_sample(2 * teta_count) * 2 * pi
    Result_min = minimize(function, Init_params, method='COBYLA', options={
        'maxiter': 5000}, tol=10 ** (-10))
    Optimal_angles = Result_min.x
    np.save(f"Trained_params/result_{i}", Optimal_angles)
    print(Optimal_angles)
    print(Init_params)
    plt.plot(range(len(err_arr)), err_arr)
    plt.show()
