def compute_expression(mu, p, l, Rf, delta, alpha, Z_bar, k_hat):
    # Calculate the first term
    first_term = ((1 + mu - mu * p) / p) * (Z_bar * k_hat**alpha)

    # Calculate the second term
    second_term = ((l * Rf + delta * p - l * p) / p) * k_hat

    # Compute the final expression
    expression = first_term - second_term
    return expression


def compute_k_hat(Z, alpha, beta, mu, p, l, Rf, delta):
    numerator = Z * alpha * beta * (1 + mu - mu * p)
    denominator = p - p * l - beta * p + beta * delta * p + beta * Rf * l
    k_hat = (numerator / denominator) ** (1 / (1 - alpha))
    return k_hat


# Parameters
beta = 0.956
Rf = 1.04
delta = 0.07
alpha = 0.80
Z_bar = 0.2
mus = [1, 0.25]  # Derived from 1-mu
l = 0.2
p = 0.6  # Since 1-p=0
k_hat = 1  # Example capital level

# Compute the expression for each mu
for mu in mus:
    result = compute_expression(mu, p, l, Rf, delta, alpha, Z_bar, k_hat)
    print(f"Result for mu={mu}: {result}")


# Compute k_hat for each mu and then compute the expression
for mu in mus:
    k_hat = compute_k_hat(Z_bar, alpha, beta, mu, p, l, Rf, delta)
    result = compute_expression(mu, p, l, Rf, delta, alpha, Z_bar, k_hat)
    print(
        f"For mu={mu}, computed k_hat={k_hat:.4f} and the result of the expression={result:.4f}"
    )
