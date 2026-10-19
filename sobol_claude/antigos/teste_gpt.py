import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os


# ============================================================
# CONFIGURAÇÃO DO DIRETÓRIO
# ============================================================

outdir = os.path.join(
    os.path.dirname(__file__),
    "modelo_acoplado_novo_figures"
)
os.makedirs(outdir, exist_ok=True)


# ============================================================
# PARÂMETROS
# ============================================================

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

    'a': 0.2,
    'b': 0.2,
    'c': 0.2,
    'd': 0.205,
    'e': 0.2,
    'f': 2.0,
    'g': 0.2,

    'h': 2.0,
    'k': 2.0,
    'r': 1.0,
    's': 0.0,

    'H1': 1e-4,
    'H2': 1e-9,
    'H3': 1e-10,

    'B1': 1e-4,
    'B2': 1e-9,
    'B3': 1e-10
}


# ============================================================
# MODELO
# ============================================================

def coupled_model(y_vec, t, p):

    V, Ap, Apm, Thn, The, Tkn, Tke, B, Ps, Pl, Bm, A, w, x, y_inf = y_vec

    # --------------------------------------------------------
    # Proteção numérica
    # --------------------------------------------------------
    # O modelo físico trabalha somente com estados >= 0.
    # Se o integrador fizer uma pequena excursão numérica
    # para valores negativos, usamos 0 nas avaliações internas.

    Vp   = max(V, 0.0)
    App  = max(Ap, 0.0)
    Apmp = max(Apm, 0.0)
    Thnp = max(Thn, 0.0)
    Thep = max(The, 0.0)
    Tknp = max(Tkn, 0.0)
    Tkep = max(Tke, 0.0)
    Bp   = max(B, 0.0)
    Psp  = max(Ps, 0.0)
    Plp  = max(Pl, 0.0)
    Bmp  = max(Bm, 0.0)
    Apb  = max(A, 0.0)
    wp   = max(w, 0.0)
    xp   = max(x, 0.0)
    yp   = max(y_inf, 0.0)

    # --------------------------------------------------------
    # Sistema imunológico
    # --------------------------------------------------------

    # Mantida apenas para referência.
    # NÃO é utilizada na dinâmica final do vírus.
    dV_im = (
        p['πv'] * Vp
        - p['cv1'] * Vp / (p['cv2'] + Vp)
        - p['kv1'] * Vp * Apb
        - p['kv2'] * Vp * Tkep
    )

    dAp = (
        p['αap'] * (p['Ap0'] - App)
        - p['βap'] * App
        * (p['cap1'] * Vp) / (p['cap2'] + Vp)
    )

    dApm = (
        p['βap'] * App
        * (p['cap1'] * Vp) / (p['cap2'] + Vp)
        - p['δapm'] * Apmp
    )

    dThn = (
        p['αth'] * (p['Thn0'] - Thnp)
        - p['βth'] * Apmp * Thnp
    )

    dThe = (
        p['βth'] * Apmp * Thnp
        + p['πth'] * Apmp * Thep
        - p['δth'] * Thep
    )

    dTkn = (
        p['αtk'] * (p['Tkn0'] - Tknp)
        - p['βtk'] * Apmp * Tknp
    )

    dTke = (
        p['βtk'] * Apmp * Tknp
        + p['πtk'] * Apmp * Tkep
        - p['δtk'] * Tkep
    )

    dB = (
        p['αb'] * (p['B0'] - Bp)
        + p['πb1'] * Vp * Bp
        + p['πb2'] * Thep * Bp
        - p['βps'] * Apmp * Bp
        - p['βpl'] * Thep * Bp
        - p['βbm'] * Thep * Bp
    )

    dPs = (
        p['βps'] * Apmp * Bp
        - p['δps'] * Psp
    )

    dPl = (
        p['βpl'] * Thep * Bp
        - p['δpl'] * Plp
        + p['γbm'] * Bmp
    )

    dBm = (
        p['βbm'] * Thep * Bp
        + p['πbm1'] * Bmp * (1 - Bmp / p['πbm2'])
        - p['γbm'] * Bmp
    )

    dA = (
        p['πps'] * Psp
        + p['πpl'] * Plp
        - p['δa'] * Apb
    )

    # --------------------------------------------------------
    # Acoplamento imunidade -> dano / inflamação
    # --------------------------------------------------------

    hw = (
        p['H1'] * Apb
        + p['H2'] * Tkep
        + p['H3'] * App
    )

    bw = (
        p['B1'] * Apb
        + p['B2'] * Tkep
        + p['B3'] * App
    )

    dw = (
        p['s']
        + p['f'] * yp
        - p['k'] * wp
    )

    dx = (
        bw
        + p['c'] * Vp
        - p['g'] * xp
    )

    dy = (
        p['a'] * xp
        - (p['d'] + p['e'] * yp) * yp
    )

    # --------------------------------------------------------
    # Dinâmica do vírus ACOPLADA
    # --------------------------------------------------------

    dV = (
        p['r'] * Vp
        - hw * Vp
    )

    dydt = np.array([
        dV,
        dAp,
        dApm,
        dThn,
        dThe,
        dTkn,
        dTke,
        dB,
        dPs,
        dPl,
        dBm,
        dA,
        dw,
        dx,
        dy
    ], dtype=float)

    # --------------------------------------------------------
    # Garantia de invariância numérica no contorno
    # --------------------------------------------------------
    #
    # Se uma variável está em zero e uma aproximação numérica
    # produz uma derivada negativa, essa derivada é zerada.
    #
    # Isto NÃO altera a dinâmica no interior do domínio físico.
    # Atua apenas como proteção no contorno y_i = 0.

    y_original = np.array([
        V, Ap, Apm, Thn, The,
        Tkn, Tke, B, Ps, Pl,
        Bm, A, w, x, y_inf
    ])

    mask = (y_original <= 0.0) & (dydt < 0.0)

    dydt[mask] = 0.0

    return dydt


# ============================================================
# CONDIÇÕES INICIAIS
# ============================================================

y0 = np.array([
    1.0,        # V
    1e6,        # Ap
    0.0,        # Apm
    1e6,        # Thn
    0.0,        # The
    5e5,        # Tkn
    0.0,        # Tke
    2.5e5,      # B
    0.0,        # Ps
    0.0,        # Pl
    0.0,        # Bm
    150.0,      # A
    0.1,        # w
    0.0,        # x
    0.0         # y_inf
], dtype=float)


# ============================================================
# ESCALAS DE NORMALIZAÇÃO
# ============================================================

# O solver trabalha com z = y / scales.
#
# Isso evita que uma variável da ordem de 1e6 e outra da ordem
# de 1 sejam tratadas diretamente na mesma escala numérica.

scales = np.array([
    1.0,       # V
    1e6,       # Ap
    1e5,       # Apm
    1e6,       # Thn
    1e5,       # The
    5e5,       # Tkn
    1e5,       # Tke
    2.5e5,     # B
    1e5,       # Ps
    1e5,       # Pl
    1e4,       # Bm
    1e4,       # A
    1.0,       # w
    1.0,       # x
    1.0        # y_inf
], dtype=float)


# ============================================================
# MODELO NORMALIZADO PARA ODEINT
# ============================================================

def scaled_model(z, t, p):

    # Retorna ao espaço físico
    y = z * scales

    # Calcula as derivadas no espaço físico
    dy = coupled_model(y, t, p)

    # Retorna para o espaço normalizado
    dz = np.asarray(dy) / scales

    return dz


# ============================================================
# INTEGRAÇÃO
# ============================================================

if __name__ == "__main__":

    t = np.linspace(0, 500, 50000)

    # Estado inicial normalizado
    z0 = y0 / scales

    sol_z = odeint(
        scaled_model,
        z0,
        t,
        args=(params,),
        rtol=1e-8,
        atol=1e-10,
        hmax=0.05,
        mxstep=10000
    )

    # Volta para unidades físicas
    sol = sol_z * scales

    # --------------------------------------------------------
    # Proteção final contra erros numéricos residuais
    # --------------------------------------------------------
    #
    # Não deveria ser necessária com a integração acima.
    # É apenas uma última proteção contra números extremamente
    # pequenos como -1e-14 causados por arredondamento.

    sol[np.abs(sol) < 1e-5] = 0.0
    sol = np.maximum(sol, 0.0)

    # ========================================================
    # SEPARAÇÃO DAS VARIÁVEIS
    # ========================================================

    (
        V, Ap, Apm, Thn, The,
        Tkn, Tke, B, Ps, Pl,
        Bm, A, w, x, y_inf
    ) = sol.T

    # ========================================================
    # VERIFICAÇÃO
    # ========================================================

    names = [
        "V", "Ap", "Apm", "Thn", "The",
        "Tkn", "Tke", "B", "Ps", "Pl",
        "Bm", "A", "w", "x", "y_inf"
    ]

    print("\n===== VERIFICAÇÃO DE NÃO-NEGATIVIDADE =====")

    for name, data in zip(names, sol.T):
        print(
            f"{name:5s} -> "
            f"mínimo = {np.min(data):.6e}"
        )

    print(
        "\nMaior valor negativo encontrado:",
        np.min(sol)
    )

    # ========================================================
    # GRÁFICOS
    # ========================================================

    fig, axes = plt.subplots(
        3, 2,
        figsize=(12, 10)
    )

    # Linha 1
    axes[0, 0].plot(t, V, color='purple')
    axes[0, 0].set_title("Vírus / Patógeno (V)")
    axes[0, 0].set_ylabel("Carga Viral")

    axes[0, 1].plot(t, w, color='blue')
    axes[0, 1].set_title("Imunidade Inata (w)")
    axes[0, 1].set_ylabel("Nível de Ativação")

    # Linha 2
    axes[1, 0].plot(t, x, color='red')
    axes[1, 0].set_title("Dano Tecidual (x)")
    axes[1, 0].set_ylabel("Dano")

    axes[1, 1].plot(t, y_inf, color='orange')
    axes[1, 1].set_title("Inflamação (y)")
    axes[1, 1].set_ylabel("Citocinas")

    # Linha 3
    axes[2, 0].plot(t, A, color='black')
    axes[2, 0].set_title("Anticorpos (A)")
    axes[2, 0].set_ylabel("Título")

    axes[2, 1].plot(t, Ap, color='brown')
    axes[2, 1].set_title("Células apresentadoras (Ap)")
    axes[2, 1].set_ylabel("Nível de Ativação")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            outdir,
            "Matsuo_Oscillation_Simulation.png"
        ),
        dpi=200
    )

    plt.show()