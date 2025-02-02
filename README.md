# Variational Quantum Factorization

## Overview
This repository contains a variational (optimization-based) implementation for integer factorization using Qiskit. Instead of using Shor's algorithm, this approach encodes candidate factors into a bitstring, uses a parameterized quantum circuit (ansatz) to generate a probability distribution over possible factors, and then minimizes the cost function:

Cost=(N-(p*q))^2

where the output bitstring is split into two parts corresponding to the binary representations of factors \(p\) and \(q\). This method is promising for near-term quantum devices (NISQ) due to its shallow circuit depth.

## Files
- **variational_factorization.py**: The main Python script implementing the variational quantum factoring algorithm.
- **README.md**: This documentation file.

## How It Works
1. **Ansatz Construction:**  
   The circuit applies parameterized RY rotations on each qubit followed by a chain of CNOT gates to create entanglement.
2. **Cost Function:**  
   The output bitstring is split into two parts—one for factor \(p\) and one for factor \(q\). The cost is calculated as the expectation value of \((N - p \times q)^2\).
3. **Optimization:**  
   A classical optimizer (COBYLA) is used to minimize the cost function and find the optimal circuit parameters.
4. **Result Extraction:**  
   The most frequent measurement outcome is used to infer the factors.

## How to Run
1. Install Qiskit (if not already installed):
   ```bash
   pip install qiskit
2. I didnt have enough time and the qbraid platform had some issues for me..( I used the code for the 128 qubits and not the 200 qubit code given for IQuHACK
3. This approach scales linearly with the total bit length allocated. For larger semiprime numbers (e.g., 20-bit or 30-bit numbers), the required qubits will increase accordingly—demonstrating potential newsworthy benchmarks on an actual QPU
