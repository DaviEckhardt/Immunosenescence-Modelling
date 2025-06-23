import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os


pasta_destino = os.path.join(os.path.dirname(__file__), "Images for variation and after ten years")

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



valores_virus = [200, 500, 724, 1000, 1234, 7522]
cores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']


t1 = np.concatenate([
    np.linspace(0, 60, 300, endpoint=False),
    np.linspace(60, 3650, 700)
])

t2 = np.linspace(0, 60, 300)

# Condições iniciais
# y0 = [724, 1e6, 0, 1e6, 0, 5e5, 0, 2.5e5, 0, 0, 0, 150]


# # Tempo para chegar aos 10 anos (sem reforço)
# t1 = np.concatenate([
#     np.linspace(0, 60, 300, endpoint=False),       # 60 primeiros dias
#     np.linspace(60, 3650, 700)                     # resto do tempo
# ])

# # Tempo de simulação (dias)
# t2 = np.linspace(0, 60, 300)


# # Resolver o sistema
# sol1 = odeint(immune_response, y0, t1, args=(params,))

# # Preparando para a reinjeção
# y_reinj = sol1[-1].copy()
# y_reinj[0] += 724

# sol2 = odeint(immune_response, y_reinj, t2, args=(params,))


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
solucoes_primeira_dose = []
solucoes_segunda_dose = []


for V0 in valores_virus:
    y0 = [V0, 1e6, 0, 1e6, 0, 5e5, 0, 2.5e5, 0, 0, 0, 150]
    sol1 = odeint(immune_response, y0, t1, args=(params,))
    sol1_cut = sol1[t1 < 60]

    y_reinj = sol1[-1].copy()
    y_reinj[0] += V0  # Reinjetar o mesmo valor de vírus


    # Simulação pós-reforço
    sol2 = odeint(immune_response, y_reinj, t2, args=(params,))
    
    solucoes_primeira_dose.append(sol1_cut)
    solucoes_segunda_dose.append(sol2)

for i, chave in enumerate(chaves):
    # Gráfico da 1ª dose
    plt.figure(figsize=(8, 5))
    for j, sol in enumerate(solucoes_primeira_dose):
        plt.plot(t2, sol[:, i], label=f'V0 = {valores_virus[j]}', color=cores[j])
    plt.title(f"{labels[chave]} – 1ª vacinação (60 dias)")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Quantidade")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_destino, f"{chave}_primeira_variacao.png"), dpi=150)
    plt.show()

    # Gráfico do reforço
    plt.figure(figsize=(8, 5))
    for j, sol in enumerate(solucoes_segunda_dose):
        plt.plot(t2, sol[:, i], label=f'V0 = {valores_virus[j]}', color=cores[j])
    plt.title(f"{labels[chave]} – reforço após 10 anos (60 dias)")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Quantidade")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_destino, f"{chave}_reforco_variacao.png"), dpi=150)
    plt.show()





# # Índices onde t1 <= 60
# idx_t1_60 = t1 <= 60
# t1_cut = t1[idx_t1_60]
# sol1_cut = sol1[idx_t1_60]

# for i, chave in enumerate(chaves):
#     # Vacinação inicial
#     plt.figure(figsize=(7, 4))
#     plt.plot(t1_cut, sol1_cut[:, i], label=f"{labels[chave]} (vacinação inicial)", color='tab:blue')
#     plt.title(f"{labels[chave]} - 1ª vacinação (60 dias)")
#     plt.xlabel("Tempo (dias)")
#     plt.ylabel("Quantidade")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(pasta_destino, f"{chave}_primeira_60dias.png"), dpi=150)
#     plt.show()

#     # Reforço vacinal 
#     plt.figure(figsize=(7, 4))
#     plt.plot(t2, sol2[:, i], label=f"{labels[chave]} (reforço)", color='tab:green')
#     plt.title(f"{labels[chave]} - Reforço após 10 anos (60 dias)")
#     plt.xlabel("Tempo (dias)")
#     plt.ylabel("Quantidade")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(pasta_destino, f"{chave}_reforco_60dias.png"), dpi=150)
#     plt.show()