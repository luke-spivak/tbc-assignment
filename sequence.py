# (1)
# Complete the sequence_calculator function, which should
# Return the n-th number of the sequence S_n, defined as:
# S_n = 3*S_(n-1) - S_(n-2), with S_0 = 0 and S_1 = 1.
# Your implementation should minimize the execution time.
#
# (2)
# Find the time complexity of the proposed solution, using
# the "Big O" notation, and explain in detail why such
# complexity is obtained, for n ranging from 0 to at least
# 100000. HINT: you are dealing with very large numbers!
#
# (3)
# Plot the execution time VS n (again, for n ranging from 0
# to at least 100000).
#
# (4)
# Confirm that the empirically obtained time complexity curve
# from (3) matches the claimed time complexity from (2) (e.g.
# by using curve fitting techniques).
import time
import matplotlib.pyplot as plt
import numpy as np


def mat_mult(A, B):
    """Multiply two 2×2 matrices A and B."""
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
    ]


def mat_pow(M, n):
    """Compute M^n by binary exponentiation in O(log n) multiplies."""
    # Initialize R = identity
    R = [[1, 0], [0, 1]]
    while n > 0:
        if n & 1:
            R = mat_mult(R, M)
        M = mat_mult(M, M)
        n >>= 1
    return R


# (1)
def sequence_calculator(n):
    """
    Return S_n for the recurrence S_n = 3 S_{n-1} - S_{n-2},
    with S_0=0, S_1=1, via matrix exponentiation.
    """
    if n < 0:
        raise ValueError("n must be greater than 0")
    if n == 0:
        return 0
    # Companion matrix
    M = [[3, -1], [1, 0]]
    # We need M^(n-1) * [S_1, S_0]^T = M^(n-1) * [1, 0]^T
    M_exp = mat_pow(M, n - 1)
    # The top-left entry of M_exp is S_n
    return M_exp[0][0]


"""
(2) Time Complexity of sequence_calculator (Matrix Exponentiation)

The solution computes M = [[3, -1
                            1, 0]]
and then raises M to the (n-1)-th power by binary xponentiation.

There are O(log n) matrix multiplications because each bit of the exponent
n−1 leads to at most one “multiply‑into‑result” plus one “square the base” step.
If n has ⌊log_2 n⌋+1 bits, you perform at most 2⌊log_2 n⌋+1 multiplications.
Thus O(log n) matrix multiplications.

By the time you reach M^n, M's numbers are about as big as S_n, 
which has on the order of n bits.

Multiplying two n-bit integers in Python takes roughly k*log(k) time 
for a k bit integer (that’s how FFT–based big‑integer math works).

So the total time complexity for sequence_calculator is O(log n) * O(nlog n), or:
O(n(log n)^2)
"""


# (3)
def measure_execution_time():
    # Measure execution time for various n
    ns = np.arange(1000, 100001, 5000)
    times = []
    for n in ns:
        runs = 3
        durations = []
        for _ in range(runs):
            start = time.perf_counter()
            sequence_calculator(n)
            durations.append(time.perf_counter() - start)
        times.append(sum(durations) / runs)

    # Plot
    plt.figure()
    plt.plot(ns, times, marker="o")
    plt.xlabel("n")
    plt.ylabel("Execution time (s)")
    plt.title("Execution time of sequence_calculator vs n")
    plt.tight_layout()
    plt.show()


# (4)
def confirm_execution_time():
    # Sample n values and measure times
    ns = np.arange(1000, 100001, 5000)
    times = []
    for n in ns:
        # Average over a few runs
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            sequence_calculator(n)
            durations.append(time.perf_counter() - start)
        times.append(np.mean(durations))
    times = np.array(times)

    # Prepare X = n * (log2(n))^2
    log_n = np.where(ns > 0, np.log2(ns), 0)
    X = ns * (log_n**2)

    # Linear regression: time ~ a * X + b
    coeffs = np.polyfit(X, times, 1)
    a, b = coeffs
    pred = a * X + b

    # Compute R^2
    ss_res = np.sum((times - pred) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Plot measured vs predicted
    plt.figure()
    plt.scatter(X, times, label="Measured")
    plt.plot(X, pred, label=f"Fit: {a:.2e}*X + {b:.2e}")
    plt.xlabel("n * (log2 n)^2")
    plt.ylabel("Execution time (s)")
    plt.title("Fit of execution time to n*(log n)^2 model")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Regression result: time = {a:.3e} * [n*(log2(n))^2] + {b:.3e}")
    print(f"R^2 = {r2:.4f}")


if __name__ == "__main__":
    # measure_execution_time()
    confirm_execution_time()
