import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# Criar pasta para salvar imagens
pasta_destino = os.path.join(os.path.dirname(__file__), "Images_aging_by_age_gender")
os.makedirs(pasta_destino, exist_ok=True)

t = np.linspace(0, 60, 300)
solucoes = {}

# ------------------------------------------------------------
# Dados dos artigos (valores aproximados)
# ------------------------------------------------------------

# IL-6 (pg/mL)
idades_il6 = [25, 45, 65, 75]
il6_homens = [0.8, 1.1, 1.6, 1.8]
il6_mulheres = [0.9, 1.4, 2.2, 2.5]

# CD4 naïve (células/µL)
idades_cd4 = [25, 45, 65]
cd4_homens = [750, 350, 120]
cd4_mulheres = [700, 300, 90]  # queda mais forte pós-menopausa

# NK (células/µL)
idades_nk = [25, 45, 65, 75]
nk_homens = [180, 250, 300, 350]
nk_mulheres = [150, 220, 320, 360]

# B cells naïve %
idades_b = [25, 45, 65, 75]
b_naive_homens = [60, 50, 40, 35]   # %
b_naive_mulheres = [65, 55, 35, 30] # %

# ------------------------------------------------------------
# Interpoladores
# ------------------------------------------------------------
interp_il6 = {
    "M": interp1d(idades_il6, il6_homens, kind="cubic", fill_value="extrapolate"),
    "F": interp1d(idades_il6, il6_mulheres, kind="cubic", fill_value="extrapolate")
}
interp_cd4 = {
    "M": interp1d(idades_cd4, cd4_homens, kind="linear", fill_value="extrapolate"),
    "F": interp1d(idades_cd4, cd4_mulheres, kind="linear", fill_value="extrapolate")
}
interp_nk = {
    "M": interp1d(idades_nk, nk_homens, kind="cubic", fill_value="extrapolate"),
    "F": interp1d(idades_nk, nk_mulheres, kind="cubic", fill_value="extrapolate")
}
interp_b = {
    "M": interp1d(idades_b, b_naive_homens, kind="linear", fill_value="extrapolate"),
    "F": interp1d(idades_b, b_naive_mulheres, kind="linear", fill_value="extrapolate")
}

# ------------------------------------------------------------
# Parâmetros do modelo
# ------------------------------------------------------------
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
    'σ_base': 1.5e-3
}

labels = [
    "V", "Ap", "Apm", "Thn", "The", "Tkn", "Tke", "B", "Ps", "Pl", "Bm", "A", "Tprod", "I", "S", "NK"
]

# ------------------------------------------------------------
# Condições iniciais dependentes de idade/sexo
# ------------------------------------------------------------
def gerar_condicoes_iniciais(idade, sexo):
    Tprod_0 = interp_cd4[sexo](idade)
    I_0 = interp_il6[sexo](idade)
    NK_0 = interp_nk[sexo](idade)
    frac_b_naive = interp_b[sexo](idade) / 100
    B_total = 2.5e5
    B_naive = frac_b_naive * B_total
    B_memory = (1 - frac_b_naive) * B_total
    S_0 = 0.4 * I_0 + 0.01 * idade

    V0 = 724
    return [
        V0, 1e6, 0, 1e6, 0, 5e5, 0, B_naive, 0, B_memory, 0, 150,
        Tprod_0, I_0, S_0, NK_0
    ]

# ------------------------------------------------------------
# Modelo ODE
# ------------------------------------------------------------
def immune_response_aging(y, t, p, idade, sexo):
    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, Tprod, I, S, NK = y

    μ_T_eff = p['μ_T'] * (1 + 0.02 * (idade - 20))
    if sexo == "F" and idade >= 50:
        μ_T_eff *= 1.2  # menopausa acelera perda tímica

    δth_eff = p['δth'] * (1 + p['γ_I'] * I + 0.005 * idade)
    dTprod = -μ_T_eff * Tprod
    dI = p['k_I'] - p['δ_I'] * I
    dS = p['σ_B'] * B + p['σ_T'] * The + p['σ_base'] * idade - p['δ_S'] * S
    dNK = 0.01 * idade - 0.001 * NK  # dinâmica simplificada

    βpl_eff = p['βpl'] * np.exp(-p['η_S'] * S)

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

    return [dV, dAp, dApm, dThn, dThe, dTkn, dTke, dB, dPs, dPl, dBm, dA, dTprod, dI, dS, dNK]

# ------------------------------------------------------------
# Rodar simulações
# ------------------------------------------------------------
idades = [25, 45, 65]
sexos = ["M", "F"]

for sexo in sexos:
    for idade in idades:
        y0 = gerar_condicoes_iniciais(idade, sexo)
        sol = odeint(lambda y, t: immune_response_aging(y, t, params, idade, sexo), y0, t)
        solucoes[(idade, sexo)] = sol

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
for i, label in enumerate(labels):
    plt.figure(figsize=(8, 5))
    for sexo in sexos:
        for idade in idades:
            estilo = '-' if sexo == "M" else '--'
            plt.plot(t, solucoes[(idade, sexo)][:, i], label=f"{sexo}, {idade}a", linestyle=estilo)
    plt.title(f"{label} – efeito da idade e sexo")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_destino, f"{label}_idade_sexo.png"), dpi=150)
    plt.show()
