"""
modelo_acoplado_novo.py
-----------------------
Modelo imunológico acoplado (15 variáveis de estado) para análise
de sensibilidade de Sobol com MPI.

Variáveis de estado (ordem do vetor y):
  0  V      Vírus / Patógeno
  1  Ap     APCs ingênuas
  2  Apm    APCs maduras
  3  Thn    T CD4+ não-ativados
  4  The    T CD4+ efetores
  5  Tkn    T CD8+ não-ativados
  6  Tke    T CD8+ efetores
  7  B      Linfócitos B
  8  Ps     Plasmócitos curta duração
  9  Pl     Plasmócitos longa duração
  10 Bm     Células B de memória
  11 A      Anticorpos               <- QoI: pico de A
  12 w      Imunidade inata
  13 x      Dano tecidual
  14 y_inf  Inflamação

Nota sobre dV_im: a equação imune do vírus é calculada internamente mas
não entra no return — o modelo usa a equação acoplada (dV = r*V - hw*V).
Isso é intencional no design do modelo acoplado e não foi alterado.

Uso standalone (plota a simulação nominal):
    python modelo_acoplado_novo.py
"""

import os
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use("Agg")          # seguro em clusters sem display
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# Parâmetros nominais
# ─────────────────────────────────────────────────────────────────
params = {
    # Subsistema imune
    'πv':   6.80e-1,
    'cv1':  2.63e0,
    'cv2':  6e-1,
    'kv1':  4.82e-5,
    'kv2':  7.48e-7,
    'αap':  2.5,        # valor correto para o modelo acoplado
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
    # Subsistema acoplado
    'a': 0.2,  'b': 0.2,  'c': 0.2,  'd': 0.205,
    'e': 0.2,  'f': 2.0,  'g': 0.2,  'h': 2.0,
    'k': 2.0,  'r': 1.0,  's': 0.0,
    'H1': 1e-4, 'H2': 1e-9,  'H3': 1e-10,
    'B1': 1e-4, 'B2': 1e-9,  'B3': 1e-10,
}

# ─────────────────────────────────────────────────────────────────
# Parâmetros amostrados pelo Sobol.
# CRÍTICO: esta lista deve ser idêntica (mesma ordem) a
# problem['names'] em sobol_pipeline_mpi.py.
# Seguindo o código de referência MPI (documento fornecido).
# ─────────────────────────────────────────────────────────────────
param_keys = [
    'αtk',   # 0
    'βtk',   # 1
    'πps',   # 2
    'πtk',   # 3
    'δps',   # 4
    'αb',    # 5
    'H1',    # 6
    'f',     # 7
    'c',     # 8
    'r',     # 9
]

# Condições iniciais (15 variáveis)
y0 = [
    1.0,    # V
    1e6,    # Ap
    0.0,    # Apm
    1e6,    # Thn
    0.0,    # The
    5e5,    # Tkn
    0.0,    # Tke
    2.5e5,  # B
    0.0,    # Ps
    0.0,    # Pl
    0.0,    # Bm
    150.0,  # A
    0.1,    # w
    0.0,    # x
    0.0,    # y_inf
]


# ─────────────────────────────────────────────────────────────────
# Sistema de EDOs acoplado
# ─────────────────────────────────────────────────────────────────
def coupled_model(y_vec, t, p):
    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, w, x, y_inf = y_vec

    # Subsistema imune (dV_im calculado mas não usado no return —
    # o modelo acoplado substitui a dinâmica do vírus por dV abaixo)
    dV_im = p['πv']*V - p['cv1']*V/(p['cv2'] + V) - p['kv1']*V*A - p['kv2']*V*Tke
    dAp   = p['αap']*(p['Ap0'] - Ap) - p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V)
    dApm  = p['βap']*Ap*(p['cap1']*V)/(p['cap2'] + V) - p['δapm']*Apm
    dThn  = p['αth']*(p['Thn0'] - Thn) - p['βth']*Apm*Thn
    dThe  = p['βth']*Apm*Thn + p['πth']*Apm*The - p['δth']*The
    dTkn  = p['αtk']*(p['Tkn0'] - Tkn) - p['βtk']*Apm*Tkn
    dTke  = p['βtk']*Apm*Tkn + p['πtk']*Apm*Tke - p['δtk']*Tke
    dB    = (p['αb']*(p['B0'] - B) + p['πb1']*V*B + p['πb2']*The*B
             - p['βps']*Apm*B - p['βpl']*The*B - p['βbm']*The*B)
    dPs   = p['βps']*Apm*B - p['δps']*Ps
    dPl   = p['βpl']*The*B - p['δpl']*Pl + p['γbm']*Bm
    dBm   = p['βbm']*The*B + p['πbm1']*Bm*(1 - Bm/p['πbm2']) - p['γbm']*Bm
    dA    = p['πps']*Ps + p['πpl']*Pl - p['δa']*A

    # Subsistema acoplado
    hw = p['H1']*A + p['H2']*Tke + p['H3']*Ap
    bw = p['B1']*A + p['B2']*Tke + p['B3']*Ap

    dw  = p['s'] + p['f']*y_inf - p['k']*w
    dx  = bw + p['c']*V - p['g']*x
    dy  = p['a']*x - (p['d'] + p['e']*y_inf)*y_inf
    dV  = p['r']*V - hw*V        # equação acoplada do vírus (substitui dV_im)

    return [dV, dAp, dApm, dThn, dThe, dTkn, dTke, dB, dPs, dPl, dBm, dA, dw, dx, dy]


# ─────────────────────────────────────────────────────────────────
# Wrapper chamado pelo pipeline MPI
# Recebe X com os valores de param_keys nesta ordem.
# QoI = pico de Anticorpos (col 11 = A).
# ─────────────────────────────────────────────────────────────────
def coupled_wrapper(X):
    X = np.atleast_1d(X).flatten()
    p = params.copy()

    for i, key in enumerate(param_keys):
        p[key] = X[i]

    t = np.linspace(0, 500, 5000)
    sol = odeint(coupled_model, y0, t, args=(p,), full_output=False)

    result = float(np.max(sol[:, 11]))   # pico de A (Anticorpos)

    if not np.isfinite(result):
        return 0.0

    return result


# ─────────────────────────────────────────────────────────────────
# Execução standalone
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "modelo_acoplado_novo_figures")
    os.makedirs(outdir, exist_ok=True)

    t = np.linspace(0, 500, 50000)
    sol = odeint(lambda y, t: coupled_model(y, t, params), y0, t)

    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, w, x, y_inf = sol.T

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].plot(t, V, color='purple');   axes[0, 0].set_title("Vírus / Patógeno (V)");    axes[0, 0].set_ylabel("Carga Viral")
    axes[0, 1].plot(t, w, color='blue');     axes[0, 1].set_title("Imunidade Inata (w)");     axes[0, 1].set_ylabel("Nível de Ativação")
    axes[1, 0].plot(t, x, color='red');      axes[1, 0].set_title("Dano Tecidual (x)");       axes[1, 0].set_ylabel("Dano")
    axes[1, 1].plot(t, y_inf, color='orange'); axes[1, 1].set_title("Inflamação (y)");          axes[1, 1].set_ylabel("Citocinas")
    axes[2, 0].plot(t, A, color='black');    axes[2, 0].set_title("Anticorpos (A)");           axes[2, 0].set_ylabel("Título")
    axes[2, 1].plot(t, Ap, color='brown');   axes[2, 1].set_title("Células apresentadoras (Ap)"); axes[2, 1].set_ylabel("Quantidade")

    for ax in axes.flat:
        ax.set_xlabel("Tempo (dias)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Matsuo_Oscillation_Simulation.png"), dpi=200)
    plt.close(fig)
    print(f"[OK] Figura salva em: {outdir}")

    # Verificação rápida do wrapper
    X_nom = np.array([params[k] for k in param_keys])
    qoi = coupled_wrapper(X_nom)
    print(f"[OK] Wrapper com nominais — pico de Anticorpos: {qoi:.4e}")
