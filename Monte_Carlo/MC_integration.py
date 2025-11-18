# int_a^b f(x)dx = (b-a) * E[f(X)] where X ~ Uniform(a, b)
# E[f(X)] = int_a^b f(x) * (1/(b-a)) dx

import random
class MonteCarloIntegrator:
    def __init__(self, a, b, n=1e5, k=1e1, seed=None):
        # sorge dafür, dass n und k ganzzahlig sind für range()
        self.a = float(a)
        self.b = float(b)
        self.n = int(n)
        self.k = int(k)
        if seed is not None:
            random.seed(seed)

    def integrate(self, integrand):
        width = self.b - self.a
        samples = []
        for _ in range(self.k):
            s = 0.0
            for _ in range(self.n):
                x = random.uniform(self.a, self.b)
                s += width * integrand(x)
            samples.append(s / self.n)
        return sum(samples) / self.k