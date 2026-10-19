"""
carla_corrigido.py
------------------
Modelo imunológico corrigido para análise de sensibilidade de Sobol com MPI.

Este arquivo define o modelo ODE e o wrapper que o pipeline MPI chama.
NÃO importa mpi4py — é puro modelo, sem acoplamento com paralelismo.

Correções aplicadas:
  1. αap = 2.5e-3  (mantido: valor do carla.py original para o modelo simples;
                    o valor 2.5 é do modelo_acoplado_novo.py, que é diferente)
  2. param_keys alinhado 1-a-1 com problem['names'] do pipeline Sobol
  3. QoI = np.max(sol[:, 11]) — pico de Anticorpos (A), não pico de Vírus

Uso standalone (gera e plota a simulação nominal):
    python carla_corrigido.py
"""

import os
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use("Agg")          # sem display — seguro em clusters
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# Parâmetros nominais do modelo
# ─────────────────────────────────────────────────────────────────
params = {
    'πv':   6.80e-1,
    'cv1':  2.63e0,
    'cv2':  6e-1,
    'kv1':  4.82e-5,
    'kv2':  7.48e-7,
    'αap':  2.5e-3,    # valor do carla.py original (modelo imune simples)
    'βap':  5.5e-1,
    'cap1': 8e-1,
    'cap2': 4e1,
    'δapm': 5.38e-1,
    'αth':  2.17e-4,
    'βth':  1e-7,
    'πth':  1e-8,
    'δth':  2.2e-1,
    'αtk':  2.17e-4,
    'βtk':  1e-5,
    'πtk':  1e-8,
    'δtk':  3e-4,
    'αb':   6.0e0,
    'πb1':  4.83e-6,
    'πb2':  1.27e-8,
    'βps':  6.72e-4,
    'βpl':  5.61e-6,
    'βbm':  1e-6,
    'δps':  2.0e0,
    'δpl':  2.4e-4,
    'γbm':  9.75e-4,
    'πbm1': 1e-5,
    'πbm2': 2.5e3,
    'πps':  2e-3,
    'πpl':  6.8e-4,
    'δa':   4e-2,
    'Ap0':  1e6,
    'Thn0': 1e6,
    'Tkn0': 5e5,
    'B0':   2.5e5,
}

# ─────────────────────────────────────────────────────────────────
# Parâmetros amostrados pelo Sobol.
# CRÍTICO: esta lista deve ser idêntica (mesma ordem) a
# problem['names'] em sobol_pipeline_mpi.py.
# ─────────────────────────────────────────────────────────────────
param_keys = [
    'πv',    # 0
    'kv1',   # 1
    'kv2',   # 2
    'αb',    # 3
    'βps',   # 4
    'βpl',   # 5
    'βbm',   # 6
    'δpl',   # 7
    'B0',    # 8
    'πb2',   # 9
]

# Condições iniciais
y0 = [724.0, 1e6, 0.0, 1e6, 0.0, 5e5, 0.0, 2.5e5, 0.0, 0.0, 0.0, 150.0]


# ─────────────────────────────────────────────────────────────────
# Sistema de EDOs
# ─────────────────────────────────────────────────────────────────
def immune_response(y, t, p):
    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A = y

    dV   = p['πv']*V - p['cv1']*V/(p['cv2'] + V) - p['kv1']*V*A - p['kv2']*V*Tke
    dAp  = p['αap']*(p['Ap0'] - Ap) - p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V)
    dApm = p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V) - p['δapm']*Apm
    dThn = p['αth']*(p['Thn0'] - Thn) - p['βth']*Apm*Thn
    dThe = p['βth']*Apm*Thn + p['πth']*Apm*The - p['δth']*The
    dTkn = p['αtk']*(p['Tkn0'] - Tkn) - p['βtk']*Apm*Tkn
    dTke = p['βtk']*Apm*Tkn + p['πtk']*Apm*Tke - p['δtk']*Tke
    dB   = (p['αb']*(p['B0'] - B) + p['πb1']*V*B + p['πb2']*The*B
            - p['βps']*Apm*B - p['βpl']*The*B - p['βbm']*The*B)
    dPs  = p['βps']*Apm*B - p['δps']*Ps
    dPl  = p['βpl']*The*B - p['δpl']*Pl + p['γbm']*Bm
    dBm  = p['βbm']*The*B + p['πbm1']*Bm*(1 - Bm/p['πbm2']) - p['γbm']*Bm
    dA   = p['πps']*Ps + p['πpl']*Pl - p['δa']*A

    return [dV, dAp, dApm, dThn, dThe, dTkn, dTke, dB, dPs, dPl, dBm, dA]


# ─────────────────────────────────────────────────────────────────
# Wrapper chamado pelo pipeline MPI
# Recebe um vetor X com os valores de param_keys nesta ordem.
# Retorna um escalar: pico de Anticorpos (col 11 = A).
# ─────────────────────────────────────────────────────────────────
def immune_response_wrapper(X):
    X = np.atleast_1d(X).flatten()
    p = params.copy()

    for i, key in enumerate(param_keys):
        p[key] = X[i]

    t = np.linspace(0, 500, 5000)
    sol = odeint(immune_response, y0, t, args=(p,), full_output=False)

    result = float(np.max(sol[:, 11]))   # pico de A (Anticorpos)

    if not np.isfinite(result):
        return 0.0

    return result


# ─────────────────────────────────────────────────────────────────
# Execução standalone
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pasta_destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
    os.makedirs(pasta_destino, exist_ok=True)

    t = np.linspace(0, 500, 50000)
    sol = odeint(immune_response, y0, t, args=(params,))

    labels = {
        "V":   "Vírus vacinal",
        "Ap":  "APCs ingênuas",
        "Apm": "APCs maduras",
        "Thn": "T CD4+ não-ativados",
        "The": "T CD4+ efetores",
        "Tkn": "T CD8+ não-ativados",
        "Tke": "T CD8+ efetores",
        "B":   "Linfócitos B",
        "Ps":  "Plasmócitos curta duração",
        "Pl":  "Plasmócitos longa duração",
        "Bm":  "Células B de memória",
        "A":   "Anticorpos",
    }

    for i, (chave, nome) in enumerate(labels.items()):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, sol[:, i], color="tab:blue")
        ax.set_title(nome)
        ax.set_xlabel("Tempo (dias)")
        ax.set_ylabel("Quantidade")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(pasta_destino, f"{chave}.png"), dpi=150)
        plt.close(fig)

    print(f"[OK] Figuras salvas em: {pasta_destino}")

    # Verificação rápida do wrapper com parâmetros nominais
    X_nom = np.array([params[k] for k in param_keys])
    qoi = immune_response_wrapper(X_nom)
    print(f"[OK] Verificação wrapper — pico de Anticorpos: {qoi:.4e}")
