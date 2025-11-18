import numpy as np
# M: Diskretisierungspunkte, N: Anzahl simulierter Pfade

def mc_call(S0, K, r, sigma, T, N=1e5):
    Z = np.random.normal(0, 1,int(N)) 
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.exp(-r*T) * np.maximum(ST - K, 0)
    return payoff.mean()

def mc_call_euler(S0, r, K, sigma, T, N=1e5, M=252):
    dt = T / int(M)
    S = np.zeros(M+1)
    payoff = np.zeros(int(N))
    S[0] = S0
    Z = np.random.normal(0, 1, (int(N), int(M))) 
    for i in range(int(N)): # für jeden Pfad
        for j in range(1, int(M)+1): # für jeden Zeitschritt
            S[j] = S[j-1] + r * S[j-1] * dt + sigma * S[j-1] * np.sqrt(dt) * Z[i,j-1]
        payoff[i] = np.exp(-r*T) * np.maximum(S[M] - K, 0)
    return payoff.mean()

# Beispiel:
preis = mc_call(S0=100, K=100, r=0.03, sigma=0.2, T=1.0, N=1e5)
print(f"Preis mittels expliziter Formel  ={preis:.4f}")

# Beispiel:
preis = mc_call_euler(S0=100, K=100, r=0.03, sigma=0.2, T=1.0, N=1e4)
print(f"Preis mittels Euler-Maruyama  ={preis:.4f}")

