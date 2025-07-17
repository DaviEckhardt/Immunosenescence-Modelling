
import numpy as np
import matplotlib.pyplot as plt
import os

# Pasta para salvar os gráficos
pasta_destino = os.path.join(os.path.dirname(__file__), "Images_estacionario")
os.makedirs(pasta_destino, exist_ok=True)

# Funções dependentes da idade (a em anos)
def alpha_T(a):  # Produção de linfócitos T_v
    return 1e5 * np.exp(-0.05 * np.maximum(a - 20, 0))

def delta_T(a):  # Morte/apoptose de T_v
    return 0.01 + 0.002 * np.maximum(a - 20, 0)

def beta():  # Taxa de diferenciação T_v → T_m
    return 0.01

def delta_m():  # Morte de T_m
    return 0.01

def gamma_B():  # Ativação de B por T_m
    return 0.02

def delta_B():  # Morte de B
    return 0.01

def kappa():  # Produção de anticorpos por B
    return 0.1

def delta_A():  # Degradação de anticorpos
    return 0.05

def eta(a):  # Produção de IL-6 (inflammaging)
    return 0.01 * np.maximum(a - 20, 0)

def mu_I():  # Regulação de IL-6
    return 0.01

def NK(a):  # Natural Killers
    NK0 = 1e4
    k_NK = 0.03
    return NK0 * np.exp(-k_NK * a)

def DC(a):  # Células Dendríticas
    DC0 = 5000
    r_DC = 0.01
    val = DC0 * (1 - r_DC * a)
    return np.maximum(val, 0)

def Tel(a):  # Telômeros
    Tel0 = 8.0  # unidades arbitrárias
    r_Tel = 0.05
    val = Tel0 - r_Tel * a
    return np.maximum(val, 0)

# Faixa de idades
idades = np.linspace(0, 100, 200)

# Inicializar variáveis
T_v = alpha_T(idades) / (delta_T(idades) + beta())
T_m = beta() * T_v / delta_m()
B = gamma_B() * T_m / delta_B()
A = kappa() * B / delta_A()
I = eta(idades) / mu_I()
NK_vals = NK(idades)
DC_vals = DC(idades)
Tel_vals = Tel(idades)

# Plots
variaveis = {
    "T_v (T virgens)": T_v,
    "T_m (T memória)": T_m,
    "B (Células B)": B,
    "A (Anticorpos)": A,
    "I (IL-6)": I,
    "NK (Natural Killers)": NK_vals,
    "DC (Dendríticas)": DC_vals,
    "Tel (Telômeros)": Tel_vals,
}

for nome, valores in variaveis.items():
    plt.figure(figsize=(8, 4))
    plt.plot(idades, valores, label=nome)
    plt.title(f"{nome} vs Idade")
    plt.xlabel("Idade (anos)")
    plt.ylabel(nome)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_destino, f"{nome.split()[0]}.png"), dpi=150)
    plt.close()
