import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

outdir = os.path.join(os.path.dirname(__file__), "modelo_acoplado_figures_Ap")
os.makedirs(outdir, exist_ok=True)

params = {
    'πv': 6.80e-1,
    'cv1': 2.63e0,
    'cv2': 6e-1,
    'kv1': 4.82e-5,
    'kv2': 7.48e-7,
    'αap': 2.5,
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

    'a': 0.2, 'b': 0.2, 'c': 0.2, 'd': 0.205, 'e': 0.2, 'f': 8.0, 'g': 0.2,
    'h': 2.0, 'k': 2.0, 'r': 1.0, 's': 0.0, 

    'H1': 1e-4, 'H2': 1e-7, 'H3': 1e-8,  
    'B1': 1e-4, 'B2': 1e-7, 'B3': 1e-8  
}

def g_time(t):
    return 0.2/ (1 + 0.05 * t) 

def f_time(t):
    return 8.0 / (1 + 0.2 * t)

def d_time(t):
    return 0.08 / (1 + 0.5 * t)

def k(t):
    return params['k'] / (1 + 1e-3 * t)

def coupled_model(y, t, p):

    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, w, x, y_inf  = y

    dV = 0
    dV_im = p['πv']*V - p['cv1']*V/(p['cv2'] + V) - p['kv1']*V*A - p['kv2']*V*Tke
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

    hw = p['H1'] * A + p['H2'] * Tke + p['H3'] * Ap
    bw = p['B1'] * A + p['B2'] * Tke + p['B3'] * Ap
    
    dw = p['s'] + f_time(t) * y_inf - k(t) * w
    dx = bw + p['c'] * V - g_time(t) * x
    dy = p['a'] * x - (d_time(t) + p['e'] * y_inf) * y_inf

    dV = (p['r'] * V - hw * V)
    #dV = (p['r'] * V - hw * V) + dV_im

    return [dV, dAp, dApm, dThn, dThe, dTkn, dTke, dB, dPs, dPl, dBm, dA, dw, dx, dy]


y0 = [
    724.0,      # V (antígeno inicial)
    1e6,       # Ap (células apresentadoras ingênuas)
    0,          # Apm (células apresentadoras maduras)
    1e6,        # Thn (células T helper ingênuas)
    0,          # The (células T helper efetoras)   
    5e5,        # Tkn (células T citotóxicas ingênuas)
    0,          # Tke (células T citotóxicas efetoras)
    2.5e5,      # B (células B)
    0,          # Ps (plasmablastos)
    0,          # Pl (plasmócitos)
    0,          # Bm (células B de memória)
    150,        # A (anticorpos)
    0.1,       # w (resposta imune inicial)
    0.0,        # x (dano tecidual inicial)
    0.0,        # y_inf (inflamação inicial)
]

t = np.linspace(0, 400, 801)


sol = odeint(lambda y, t: coupled_model(y, t, params), y0, t)

V, Ap, Apm, dThn, dThe, Tkn, Tke, dB, dPs, dPl, dBm, A ,w, x, y_inf = sol.T

fig2, axes2 = plt.subplots(3, 2, figsize=(10, 7))
axes2[0,0].plot(t, A); axes2[0,0].set_title("Anticorpos (A)")
axes2[0,1].plot(t, w); axes2[0,1].set_title("w (resposta imune)")
axes2[1,0].plot(t, x); axes2[1,0].set_title("x (dano tecidual)")
axes2[1,1].plot(t, Ap); axes2[1,1].set_title("Células Apresentadoras ingênuas (Ap)")
axes2[2,0].plot(t, y_inf); axes2[2,0].set_title("Citocinas pró-inflamatórias (y_inf)")
axes2[2,1].plot(t, V); axes2[2,1].set_title("Vírus vacinal (V)")
for ax in axes2.flat:
    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Nível relativo")
    ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "Teste.png"), dpi=200)
plt.show()


