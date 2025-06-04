import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os


pasta_destino = os.path.join(os.path.dirname(__file__), "Images")

os.makedirs(pasta_destino, exist_ok=True)


params = {
    'πv': 6.80e-1,
    'cv1': 2.63e0,
    'cv2': 6e-1,
    'kv1': 4.82e-5,
    'kv2': 7.48e-7,
    'αap': 2.5e-3,
    'βap': 5.5e-1,
    'cap1': 8e-1,
    'cap2': 4e1,
    'δapm': 5.38e-1,
    'αth': 2.17e-4,
    'βth': 1e-7,
    'πth': 1e-8,
    'δth': 2.2e-1,
    'αtk': 2.17e-4,
    'βtk': 1e-5,
    'πtk': 1e-8,
    'δtk': 3e-4,
    'αb': 6.0e0,
    'πb1': 4.83e-6,
    'πb2': 1.27e-8,
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
    'Ap0': 1e6,
    'Thn0': 1e6,
    'Tkn0': 5e5,
    'B0': 2.5e5,
}



def immune_response(y, t, p):
    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A = y

    dV = p['πv']*V - p['cv1']*V/(p['cv2'] + V) - p['kv1']*V*A - p['kv2']*V*Tke
    dAp = p['αap']*(p['Ap0'] - Ap) - p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V)
    dApm = p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V) - p['δapm']*Apm
    dThn = p['αth']*(p['Thn0'] - Thn) - p['βth']*Apm*Thn
    dThe = p['βth']*Apm*Thn + p['πth']*Apm*The - p['δth']*The
    dTkn = p['αtk']*(p['Tkn0'] - Tkn) - p['βtk']*Apm*Tkn
    dTke = p['βtk']*Apm*Tkn + p['πtk']*Apm*Tke - p['δtk']*Tke
    dB = (p['αb'] * (p['B0'] - B) + p['πb1'] * V * B + p['πb2'] * The * B - p['βps'] * Apm * B - p['βpl'] * The * B - p['βbm'] * The * B)
    dPs = p['βps']*Apm*B - p['δps']*Ps
    dPl = p['βpl']*The*B - p['δpl']*Pl + p['γbm']*Bm
    dBm = p['βbm']*The*B + p['πbm1']*Bm*(1 - Bm/p['πbm2']) - p['γbm']*Bm
    dA = p['πps']*Ps + p['πpl']*Pl - p['δa']*A

    return [dV, dAp, dApm, dThn, dThe, dTkn, dTke, dB, dPs, dPl, dBm, dA]

# Condições iniciais
y0 = [724, 1e6, 0, 1e6, 0, 5e5, 0, 2.5e5, 0, 0, 0, 150]

# Tempo de simulação (dias)
t = np.linspace(0, 60, 300)

# Resolver o sistema
sol = odeint(immune_response, y0, t, args=(params,))

# Plotar os resultados
labels = {
    "V": "Vírus vacinal",
    "Ap": "Células Apresentadoras de Antígeno - ingênuas",
    "Apm": "Células Apresentadoras de Antígeno - maduras",
    "Thn": "Linfócitos T CD4+ não-ativados",
    "The": "Linfócitos T CD4+ efetores",
    "Tkn": "Linfócitos T CD8+ não-ativados",
    "Tke": "Linfócitos T CD8+ efetores",
    "B": "Linfócitos B",
    "Ps": "Plasmócitos de vida curta",
    "Pl": "Plasmócitos de vida longa",
    "Bm": "Células B de memória",
    "A": "Anticorpos"
}

chaves = list(labels.keys())
for i, chave in enumerate(chaves):
    plt.figure(figsize=(7, 4))
    plt.plot(t, sol[:, i], label=labels[chave], color='tab:blue')
    plt.title(labels[chave])
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Quantidade")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    caminho_arquivo = os.path.join(pasta_destino, f"{chave}.png")
    plt.savefig(caminho_arquivo, dpi=150)
    plt.show()
