import math


def N(r_f, delta, beta, p, Delta):
    # Compute the square root of Delta
    sqrt_Delta = math.sqrt(Delta)

    # Compute the two possible values for the numerator
    numerator1 = beta * delta + Delta
    numerator2 = beta * delta - Delta

    return numerator1, numerator2


def D(R_f, beta, p):
    # Compute the denominator
    denominator = 2 * p * (R_f * beta - 1)

    return denominator


def K(alpha, beta, R_f, r_f, delta, p, Z, N, D):
    # Calculate the first part of the equation
    k = (
        ((D / N * (1 - beta * R_f)) + beta * (r_f + delta) + delta * (1 - p) / p)
        * p
        / (Z * alpha)
    )

    return k


def E(N, D, C, k):
    # given the capital at time t+1 and C we return equity
    return N / D * k + C


def Delta(beta, delta, p, R_f, r_f):
    # First part of the expression inside the square root
    first_part = (
        -beta * delta
        - beta * p**2 * delta * (1 / p - 1)
        + p**2 * delta * (1 / p - 1)
        - beta * p * delta * (1 / p - 1)
        - beta * p * r_f
    ) ** 2

    # Second part of the expression inside the square root
    second_part = (
        -4
        * (beta * p * R_f - p)
        * (
            beta**2 * p**2 * R_f
            - beta * p**2 * R_f
            + beta**2 * p * R_f
            - beta * p**2
            + p**2
            - beta * p
        )
    )

    # Total value under the square root
    sqrt_value = first_part + second_part

    return sqrt_value


Z = 0.4  # productivity
alpha = 0.3  # scale
r_f = 0.1  # risk free
p = 0.9  # probability
beta = 0.95  # discount rate
delta = 0.2  # destruction
C = -1

R_f = r_f + 1  # gross risk free
Delta_v = Delta(beta, delta, p, R_f, r_f)
N_v = N(r_f, delta, beta, p, delta)[0]  # return only the positive
D_v = float(D(R_f, beta, p))
k = float(K(alpha, beta, R_f, r_f, delta, p, Z, N_v, D_v))
e = float(E(N_v, D_v, C, k))
b = k - e

print("Numerator: ", N_v)
print("Denominator: ", D_v)
print("Delta: ", Delta_v)
print("Capital: ", k)
print("Equity: ", e)
print("Debt", b)
