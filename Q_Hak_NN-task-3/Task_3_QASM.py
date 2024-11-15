from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from math import pi
from qiskit.qasm2 import dumps
from scipy.optimize import minimize
import pyideem


N_qubits = 4
Depth = 4
N_shots = 1000
N_params = N_qubits * Depth


# Создаем для примера случайные массивы дынных
Init_params = np.random.random_sample(2 * N_params) * 2 * pi
Data = np.random.random_sample((N_qubits, N_qubits))
Answers = 2 * np.round(np.random.random_sample(N_qubits)) - 1
Iterator = 0


def Black_box(params):
    global Iterator

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
                for i in range(-1, N_qubits - 1):
                    Qc.swap(i, i + 1)

            for i in range(N_qubits):
                n = layer * N_qubits + i
                Qc.ry(params[:N_params][n], i)

    def W_operator_x():

        for layer in range(Depth):
            if layer != 0:
                for i in range(-1, N_qubits - 1):
                    Qc.swap(i, i + 1)

            for i in range(N_qubits):
                n = layer * N_qubits + i
                Qc.rx(params[N_params:][n], i)

    error = 0

    for Vector, answer in zip(Data, Answers):

        Qr = QuantumRegister(N_qubits, name="Q")
        Qc = QuantumCircuit(Qr)

        Qc.h(range(N_qubits))
        U_operator(Vector)
        Qc.h(range(N_qubits))
        U_operator(Vector)
        W_operator_y()
        W_operator_x()

        Qc.measure_all()

        # formation of a qasm file
        qasm_str = dumps(Qc)
        file = open('task3.qasm', 'w')
        file.write(qasm_str)
        file.close()

        # running on simulator
        qasm_file = "task3.qasm"
        qc = pyideem.QuantumCircuit.loadQASMFile(str(qasm_file))
        backend = pyideem.StateVector(N_qubits)
        result = qc.execute(100, backend, noise_cfg=None, return_memory=True)
        Counts = result.counts

        cost = 0
        state_arr = list(Counts.keys())

        for state in state_arr:
            if state.count("1") % 2 == 0:
                cost += Counts[state]
        if cost / N_shots < 0.5:
            value = -1 * (1 - cost / N_shots)
        else:
            value = 1 * (cost / N_shots)

        error += abs(value - answer)
    # print(error)
    return error


Result_min = minimize(Black_box, Init_params, method='COBYLA', options={'maxiter': 1000})
Optimal_angles = Result_min.x
print(Optimal_angles)

