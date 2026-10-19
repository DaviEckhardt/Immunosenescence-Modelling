import numpy as np
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

# 1. Parâmetros base (os mesmos do seu modelo original)
base_params = {
    'πv': 6.80e-1, 'cv1': 2.63e0, 'cv2': 6e-1, 'kv1': 4.82e-5, 'kv2': 7.48e-7,
    'αap': 2.5, 'βap': 5.5e-1, 'cap1': 8e-1, 'cap2': 4e1, 'δapm': 5.38e-1,
    'αth': 2.17e-4, 'βth': 1e-7, 'πth': 1e-8, 'δth': 2.2e-1, 'αtk': 2.17e-4,
    'βtk': 1e-5, 'πtk': 1e-8, 'δtk': 3e-4, 'αb': 6.0e0, 'πb1': 4.83e-6,
    'πb2': 1.27e-8, 'βps': 6.72e-4, 'βpl': 5.61e-6, 'βbm': 1e-6, 'δps': 2.0e0,
    'δpl': 2.4e-4, 'γbm': 9.75e-4, 'πbm1': 1e-5, 'πbm2': 2.5e3, 'πps': 2e-3,
    'πpl': 6.8e-4, 'δa': 4e-2, 'Ap0': 1e6, 'Thn0': 1e6, 'Tkn0': 5e5, 'B0': 2.5e5,
    'a': 0.2, 'b': 0.2, 'c': 0.2, 'd': 0.205, 'e': 0.2, 'f': 2.0, 'g': 0.2,
    'h': 2.0, 'k': 2.0, 'r': 1.0, 's': 0.0,
    'H1': 1e-4, 'H2': 1e-9, 'H3': 1e-10,
    'B1': 1e-4, 'B2': 1e-9, 'B3': 1e-10
}

# 2. Adaptação do modelo para solve_ivp (mesmo de antes)
def coupled_model_ivp(t, y_vec, p):
    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, w, x, y_inf = y_vec
    V = max(V, 0.0)

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

# 3. Definição do Problema para o SALib
problem = {
    'num_vars': 4,
    'names': ['H1', 'H2', 'c', 'g'],
    'bounds': [
        [0.5e-4, 1.5e-4],  # Limites para H1
        [0.5e-9, 1.5e-9],  # Limites para H2
        [0.1, 0.3],        # Limites para c
        [0.1, 0.3],        # Limites para g
    ]
}

# 4. Geração das amostras (Método Sobol)
# Sem efeitos de segunda ordem: N * (num_vars + 2) simulações.
N = 1024
SEED = 2026
param_values = sobol_sample.sample(problem, N, calc_second_order=False, seed=SEED)

# 5. Avaliação do Modelo (Wrapper)
Y = np.zeros([param_values.shape[0]])
y0 = [1.0, 1e6, 0, 1e6, 0, 5e5, 0, 2.5e5, 0, 0, 0, 150, 0.1, 0.0, 0.0]
t_span = (0, 500)

print(f"Iniciando {len(param_values)} simulações...")
for i, X in enumerate(param_values):
    p = base_params.copy()
    p['H1'], p['H2'], p['c'], p['g'] = X
    
    sol = solve_ivp(coupled_model_ivp, t_span, y0, args=(p,), method='LSODA', max_step=1.0)
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError(f"Simulação {i + 1} falhou: {sol.message}")
    
    # Extrai o pico de dano tecidual (variável x, índice 13)
    dano_x = sol.y[13, :]
    Y[i] = np.max(dano_x)

# 6. Cálculo dos Índices de Sensibilidade de Sobol
Si = sobol.analyze(
    problem,
    Y,
    calc_second_order=False,
    num_resamples=1000,
    seed=SEED,
)

# Exibir as estimativas brutas e os respectivos intervalos de confiança.
print("\nÍndices de Sensibilidade de Sobol:")
print("Parâmetro | S1 (IC 95%)                 | ST (IC 95%)")
for i, name in enumerate(problem['names']):
    s1_low = Si['S1'][i] - Si['S1_conf'][i]
    s1_high = Si['S1'][i] + Si['S1_conf'][i]
    st_low = Si['ST'][i] - Si['ST_conf'][i]
    st_high = Si['ST'][i] + Si['ST_conf'][i]
    print(
        f"{name:9} | {Si['S1'][i]:.4f} [{s1_low:.4f}, {s1_high:.4f}]"
        f" | {Si['ST'][i]:.4f} [{st_low:.4f}, {st_high:.4f}]"
    )
    if s1_low <= 0.0 <= s1_high:
        print(f"          | S1 de {name} não é distinguível de zero com este N.")