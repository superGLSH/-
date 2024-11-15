from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from math import pi
from qiskit_aer import AerSimulator
from qiskit import transpile


N_qubits = 5
Depth = 3
N_shots = 100000
N_params = N_qubits * Depth

# загружаем подготовленные параметры для VQML
Data = np.load("_____________")


def Quantum_layer(params):

    def function(data):

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
                        Qc.cx(i, i + 1)

                for i in range(N_qubits):
                    n = layer * N_qubits + i
                    Qc.ry(params[:N_params][n], i)

        def W_operator_x():

            for layer in range(Depth):
                if layer != 0:
                    for i in range(-1, N_qubits - 1):
                        Qc.cx(i, i + 1)

                for i in range(N_qubits):
                    n = layer * N_qubits + i
                    Qc.rx(params[N_params:][n], i)

        values = []

        for Vector in data:

            Qr = QuantumRegister(N_qubits, name="Q")
            Qc = QuantumCircuit(Qr)

            Qc.h(range(N_qubits))
            U_operator(Vector)
            Qc.h(range(N_qubits))
            U_operator(Vector)
            W_operator_y()
            W_operator_x()

            Qc.measure_all()

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

            values.append(cost)
        return values
    return function


values_arr = []
for i in range(int(Data.shape[1] / N_qubits)):
    D = Data.T[i * N_qubits: (i + 1) * N_qubits]
    Params = np.load(f"Trained_params/result_{i}.npy")
    function = Quantum_layer(Params)
    values_arr.append(function(D))
values_arr = np.array(values_arr)
print(values_arr)
