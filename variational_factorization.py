import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import COBYLA
import QuantumRingsLib
from QuantumRingsLib import QuantumRegister, AncillaRegister, ClassicalRegister, QuantumCircuit
from QuantumRingsLib import QuantumRingsProvider
from QuantumRingsLib import job_monitor
from QuantumRingsLib import JobStatus
from matplotlib import pyplot as plt
import numpy as np
import math  
provider = QuantumRingsProvider(
    token='rings-128.1Jt2JcU1HTBIcvoKnqZpTCV3Mv4ktNFL',
    name='ep24bt014@iitdh.ac.in'
)
backend = provider.get_backend("scarlet_quantum_rings")
backend = provider.get_backend("scarlet_quantum_rings")
shots = 1024

provider.active_account()

def ansatz(params, num_qubits):
    """
     An ansatz:
    - Each qubit gets a parameterized RY rotation.
    - A chain of CNOTs entangles adjacent qubits.
    """
    qc = QuantumCircuit(num_qubits)
    # Apply RY rotations  on each qubit.
    for i in range(num_qubits):
        qc.ry(params[i], i)
    # Entangle qubits with a CNOT chain.
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc

def evaluate_cost(params, num_qubits, n_p, n_q, N, shots=1024):
    """
    Evaluate the cost function as the expectation value of (N - p*q)^2.
    The bitstring produced by the circuit is split into two parts:
      - First n_p bits: binary representation of p.
      - Next n_q bits: binary representation of q.
    """
    qc = ansatz(params, num_qubits)
    qc.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    total_shots = sum(counts.values())
    cost = 0
    for bitstr, count in counts.items():
        # Qiskit returns bitstrings in little-endian order.
        bitstr = bitstr[::-1]
        p_bits = bitstr[:n_p]
        q_bits = bitstr[n_p:]
        p_val = int(p_bits, 2)
        q_val = int(q_bits, 2)
        error = N - (p_val * q_val)
        cost += (error ** 2) * (count / total_shots)
    return cost

def variational_factorization(N, n_p, n_q, maxiter=200):
    """
    Run the vqd algorithm.
    - N: The number to factor.
    - n_p: Number of bits  for factor p.
    - n_q: Number of bits  for factor q.
    
    Returns the best factors found, the minimum cost, and measurement counts.
    """
    num_qubits = n_p + n_q
    # Initialize parameters randomly.
    params = np.random.uniform(0, 2 * np.pi, num_qubits)
    optimizer = COBYLA(maxiter=maxiter)

    # Define the cost function for the optimizer.
    def cost_function(x):
        return evaluate_cost(x, num_qubits, n_p, n_q, N)
    
    opt_result = optimizer.optimize(num_vars=num_qubits, 
                                    objective_function=cost_function, 
                                    initial_point=params)
    best_params = opt_result[0]
    best_cost = opt_result[1]
    
    # Evaluate the circuit 
    qc = ansatz(best_params, num_qubits)
    qc.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=2048)
    result = job.result()
    counts = result.get_counts()
    # Choose the most frequent outcome.
    best_bitstr = max(counts, key=counts.get)
    best_bitstr = best_bitstr[::-1]
    p_bits = best_bitstr[:n_p]
    q_bits = best_bitstr[n_p:]
    p_val = int(p_bits, 2)
    q_val = int(q_bits, 2)
    
    return p_val, q_val, best_cost, counts

if __name__ == "__main__":
    # Example: Factor N = 15
    N = 15
    n_p = 3  # Number of bits for p and q
    n_q = 3  
    p, q, cost, counts = variational_factorization(N, n_p, n_q, maxiter=200)
    print(f"Found factors: {p} and {q}")
    print(f"Cost: {cost}")
    print("Measurement counts:")
    print(counts)
