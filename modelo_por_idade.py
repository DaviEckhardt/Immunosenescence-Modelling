
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# Criar pasta para salvar imagens
pasta_destino = os.path.join(os.path.dirname(__file__), "Images_aging_by_age")
os.makedirs(pasta_destino, exist_ok=True)

# Parâmetros do modelo
params = {
    'πv': 6.80e-1,
    'cv1': 2.63e0,
    'cv2': 6e-1,
    'kv1': 4.82e-5,
    'kv2': 7.48e-7,
    'βap': 5.5e-1,
    'cap1': 8e-1,
    'cap2': 4e1,
    'δapm': 5.38e-1,
    'βth': 1e-7,
    'πth': 1e-8,
    'δth': 2.2e-1,
    'βtk': 1e-5,
    'πtk': 1e-8,
    'δtk': 3e-4,
    'βps': 6.72e-4,
    'βpl': 5.61e-6,
    'βbm': 1e-6,
    'δps': 2.0e0,
    'δpl': 2.4e-4,
    'γbm': 9.75e-4,
    'πbm1': 1e-5,
    'πbm2': 2.5e3,
    'πps': 2e-3,
    'πpl': 6.8e-4,
    'δa': 4e-2,
    'αap': 2.5e-3,
    'αb': 6.0e0,
    'πb1': 4.83e-6,
    'πb2': 1.27e-8,
    'αtk': 2.17e-4,
    'αth': 2.17e-4,
    'μ_T': 0.01,
    'k_I': 0.01,
    'δ_I': 0.005,
    'σ_B': 1e-6,
    'σ_T': 1e-6,
    'δ_S': 0.01,
    'γ_I': 2.0,
    'η_S': 1e-4,
}

labels = [
    "V", "Ap", "Apm", "Thn", "The", "Tkn", "Tke", "B", "Ps", "Pl", "Bm", "A", "Tprod", "I", "S"
]

def gerar_condicoes_iniciais(idade):
    Tprod_0 = 1e5 * np.exp(-0.05 * max(idade - 20, 0))
    I_0 = 0.01 * max(idade - 20, 0)
    S_0 = 0.001 * (idade - 20)**2 if idade > 20 else 0

    V0 = 724
    return [
        V0, 1e6, 0, 1e6, 0, 5e5, 0, 2.5e5, 0, 0, 0, 150,
        Tprod_0, I_0, S_0
    ]

def immune_response_aging(y, t, p):
    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, Tprod, I, S = y

    dTprod = -p['μ_T'] * Tprod
    dI = p['k_I'] - p['δ_I'] * I
    dS = p['σ_B'] * B + p['σ_T'] * The - p['δ_S'] * S

    βpl_eff = p['βpl'] * np.exp(-p['η_S'] * S)
    δth_eff = p['δth'] * (1 + p['γ_I'] * I)

    dV = p['πv']*V - p['cv1']*V/(p['cv2'] + V) - p['kv1']*V*A - p['kv2']*V*Tke
    dAp = p['αap']*(1e6 - Ap) - p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V)
    dApm = p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V) - p['δapm']*Apm
    dThn = Tprod - p['βth']*Apm*Thn
    dThe = p['βth']*Apm*Thn + p['πth']*Apm*The - δth_eff*The
    dTkn = p['αtk']*(5e5 - Tkn) - p['βtk']*Apm*Tkn
    dTke = p['βtk']*Apm*Tkn + p['πtk']*Apm*Tke - p['δtk']*Tke
    dB = (p['αb'] * (2.5e5 - B) + p['πb1'] * V * B + p['πb2'] * The * B 
          - p['βps'] * Apm * B - βpl_eff * The * B - p['βbm'] * The * B)
    dPs = p['βps']*Apm*B - p['δps']*Ps
    dPl = βpl_eff*The*B - p['δpl']*Pl + p['γbm']*Bm
    dBm = p['βbm']*The*B + p['πbm1']*Bm*(1 - Bm/p['πbm2']) - p['γbm']*Bm
    dA = p['πps']*Ps + p['πpl']*Pl - p['δa']*A

    return [dV, dAp, dApm, dThn, dThe, dTkn, dTke, dB, dPs, dPl, dBm, dA, dTprod, dI, dS]

# Idades a simular
idades = [18, 40, 60, 80]
t = np.linspace(0, 60, 300)
solucoes = {}

for idade in idades:
    y0 = gerar_condicoes_iniciais(idade)
    sol = odeint(immune_response_aging, y0, t, args=(params,))
    solucoes[idade] = sol

# Plotar comparações por idade
for i, label in enumerate(labels):
    plt.figure(figsize=(8, 5))
    for idade in idades:
        estilo = '-' if idade <= 40 else '--'
        plt.plot(t, solucoes[idade][:, i], label=f"{idade} anos", linestyle=estilo)
    plt.title(f"{label} – efeito da idade")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_destino, f"{label}_por_idade.png"), dpi=150)
    plt.show()
