import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# Configuração de diretório para salvar imagens
outdir = os.path.join(os.path.dirname(__file__), "modelo_acoplado_dados")
os.makedirs(outdir, exist_ok=True)

# Parâmetros com as modificações aplicadas para o indivíduo idoso
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
    
    # MUDANÇA 1: Diminuir velocidade de resposta de anticorpos (era 6.0)
    'αb': 2.0e0, 
    
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
    
    # MUDANÇA 2: Produção tímica de Células T reduzida (escala de 250 cel/uL = 2.5e5)
    'Thn0': 2.5e5, 
    'Tkn0': 2.0e5, 
    
    'B0': 2.5e5,

    'a': 0.2, 'b': 0.2, 'c': 0.2, 'd': 0.205, 'e': 0.2, 'f': 2.0, 
    
    # MUDANÇA 3: Taxa de reparo tecidual diminuída (era 0.2)
    'g': 0.05, 
    
    'h': 2.0, 'k': 2.0, 'r': 1.0, 's': 0.0, 

    'H1': 1e-4, 'H2': 1e-9, 'H3': 1e-10,  
    'B1': 1e-4, 'B2': 1e-9, 'B3': 1e-10 
}

def coupled_model(y_vec, t, p):
    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, w, x, y_inf = y_vec

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

    dw = p['s'] + p['f'] * y_inf - p['k'] * w
    dx = bw + p['c'] * V - p['g'] * x
    dy = p['a'] * x - (p['d'] + p['e'] * y_inf) * y_inf
    dV = p['r'] * V - hw * V

    return [dV, dAp, dApm, dThn, dThe, dTkn, dTke, dB, dPs, dPl, dBm, dA, dw, dx, dy]

# Condições Iniciais com os dados reais mapeados
y0 = [
    1.0,        # V 
    1e6,        # Ap
    0,          # Apm
    2.5e5,      # Thn (MUDANÇA: Alinhado com a fonte de Thn0, ~250 cél/uL)
    0,          # The
    2.0e5,      # Tkn (MUDANÇA: Alinhado com a fonte de Tkn0, ~200 cél/uL)
    0,          # Tke
    2.5e5,      # B
    0,          # Ps
    0,          # Pl
    0,          # Bm
    150,        # A
    0.1,        # w 
    0.0,        # x 
    5.0,        # y_inf (MUDANÇA: Inflamação inicial de idoso, 5.0 pg/mL)
]

t = np.linspace(0, 500, 50000)

sol = odeint(lambda y, t: coupled_model(y, t, params), y0, t)
V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, w, x, y_inf = sol.T

# Plotagem
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Linha 1
axes[0,0].plot(t, V, color='purple')
axes[0,0].set_title("Vírus / Patógeno (V)")
axes[0,0].set_ylabel("Carga Viral")

axes[0,1].plot(t, w, color='blue')
axes[0,1].set_title("Imunidade Inata (w)")
axes[0,1].set_ylabel("Nível de Ativação")

axes[2,1].plot(t, Ap, color='brown')
axes[2,1].set_title("Células apresentadoras (Ap)")
axes[2,1].set_ylabel("Nível de Ativação")

# Linha 2
axes[1,0].plot(t, x, color='red')
axes[1,0].set_title("Dano Tecidual (x)")
axes[1,0].set_ylabel("Dano")

axes[1,1].plot(t, y_inf, color='orange')
axes[1,1].set_title("Inflamação (y)")
axes[1,1].set_ylabel("Citocinas")

axes[2,0].plot(t, A, color='black')
axes[2,0].set_title("Anticorpos (A)")
axes[2,0].set_ylabel("Título")

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "Matsuo_Oscillation_Simulation_Data.png"), dpi=200)
plt.show()