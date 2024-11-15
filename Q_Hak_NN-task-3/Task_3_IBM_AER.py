from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from math import pi
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.optimize import minimize
import pandas as pd
from matplotlib import pyplot as plt


N_qubits = 5
Depth = 2   # колличество применений вариационной квантовой схемы
N_shots = 1000
N_params = N_qubits * Depth

# загружаем массив векторизованных отзывов
Data = np.load("____________________")

# загружаем вектор разметки данных
Answers = pd.read_csv('___________________')
Answers["разметка"] = Answers["разметка"].apply(lambda x: 1 if x == "+" else -1)
Answers = Answers["разметка"]


def Black_box(data, answer):

    def function(params):

        def U_operator(vector):
            for i in range(N_qubits):
                Qc.rz(vector[i], i)

            for i in range(N_qubits - 1):
                Qc.cx(i, i + 1)
                Qc.rz((pi - vector[i]) * (pi - vector[i + 1]), i + 1)
                Qc.cx(i, i + 1)
                Qc.barrier()

        def W_operator_y():

            for layer in range(Depth):
                if layer != 0:
                    for k in range(N_qubits - 1):
                        Qc.cx(k, k + 1)

                for i in range(N_qubits):
                    n = layer * N_qubits + i
                    Qc.ry(params[:N_params][n], i)

        def W_operator_x():

            for layer in range(Depth):
                if layer != 0:
                    for k in range(N_qubits - 1):
                        Qc.cx(k, k + 1)

                for i in range(N_qubits):
                    n = layer * N_qubits + i
                    Qc.rx(params[N_params:][n], i)

        error = 0

        for Vector, ans in zip(data, answer):

            Qr = QuantumRegister(N_qubits, name="Q")
            Qc = QuantumCircuit(Qr)

            Qc.h(range(N_qubits))
            U_operator(Vector)
            Qc.h(range(N_qubits))
            U_operator(Vector)
            W_operator_y()
            W_operator_x()

            Qc.measure_all()

            # Qc.draw(output="mpl", initial_state=True)
            # plt.show()

            simulator = AerSimulator()
            transpiled_qc = transpile(Qc, simulator)
            job = simulator.run(transpiled_qc, shots=N_shots)
            result = job.result()
            Counts = result.get_counts()

            cost = 0
            state_arr = list(Counts.keys())

            for state in state_arr:

                if state.count("1") % 2 == 0:
                    cost += Counts[state] / N_shots
                else:
                    cost -= Counts[state] / N_shots

            error += (ans - cost)**2
        err_arr.append(error)
        return np.sqrt(error)
    return function


for i in range(int(Data.shape[1] / N_qubits)):
    err_arr = []
    D = Data.T[i * N_qubits: (i + 1) * N_qubits]
    function = Black_box(D, Answers[i * N_qubits: (i + 1) * N_qubits])
    Init_params = np.random.random_sample(2 * N_params) * 2 * pi
    Result_min = minimize(function, Init_params, method='COBYLA', options={'maxiter': 5000}, tol=10**(-10))
    Optimal_angles = Result_min.x
    np.save(f"Trained_params/result_{i}", Optimal_angles)
    print(Optimal_angles)
    print(Init_params)
    plt.plot(range(len(err_arr)), err_arr)
    plt.show()


