"""
modelo_acoplado_novo.py
-----------------------
Modelo imunológico acoplado (15 variáveis de estado) para análise
de sensibilidade de Sobol com MPI.

Variáveis de estado:
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
  11 A      Anticorpos
  12 w      Imunidade inata
  13 x      Dano tecidual
  14 y_inf  Inflamação

Modelo utilizado para o vírus:
    dV = r*V - hw*V

Observações numéricas:
- O sistema é integrado com odeint.
- As variáveis são normalizadas durante a integração.
- Avaliações internas negativas são tratadas como zero.
- Se uma variável estiver em zero e sua derivada for negativa,
  a derivada é fixada em zero.
- Na solução final, valores menores que 1e-5 são considerados zero.
- O wrapper retorna A(t) em 100 instantes para o Sobol temporal.

Wrapper:
    coupled_model_wrapper(X)

Entrada:
    X = [πv, kv1, πps, kv2, δps, αb, H1, f, c, r]

Saída:
    vetor A(t) com 100 pontos.
"""

import os
import numpy as np
from scipy.integrate import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================
# PARÂMETROS NOMINAIS
# ================================================================

params = {

    # ------------------------------------------------------------
    # Subsistema imune
    # ------------------------------------------------------------

    'πv':   6.80e-1,
    'cv1':  2.63e0,
    'cv2':  6e-1,
    'kv1':  4.82e-5,
    'kv2':  7.48e-7,

    'αap':  2.5,
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

    # ------------------------------------------------------------
    # Subsistema acoplado
    # ------------------------------------------------------------

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
    'B3': 1e-10,
}


# ================================================================
# PARÂMETROS AMOSTRADOS PELO SOBOL
#
# ESTA ORDEM DEVE SER IDÊNTICA À DO PIPELINE SOBOL.
# ================================================================

param_keys = [
    'πv',    # 0
    'kv1',   # 1
    'πps',   # 2
    'kv2',   # 3
    'δps',   # 4
    'αb',    # 5
    'H1',    # 6
    'f',     # 7
    'c',     # 8
    'r',     # 9
]


# ================================================================
# NÚMERO DE PONTOS TEMPORAIS DO WRAPPER
# ================================================================

N_TIME = 100


# ================================================================
# CONDIÇÕES INICIAIS
# ================================================================

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


# ================================================================
# ESCALAS DE NORMALIZAÇÃO
# ================================================================
#
# z_i = y_i / scales_i
#
# Evita trabalhar diretamente com variáveis que variam, por exemplo,
# entre ordem 1 e ordem 1e6.
# ================================================================

scales = np.array([
    1.0,        # V
    1e6,        # Ap
    1e5,        # Apm
    1e6,        # Thn
    1e5,        # The
    5e5,        # Tkn
    1e5,        # Tke
    2.5e5,      # B
    1e5,        # Ps
    1e5,        # Pl
    1e4,        # Bm
    1e4,        # A
    1.0,        # w
    1.0,        # x
    1.0         # y_inf
], dtype=float)


# ================================================================
# LIMIAR PARA VALORES PEQUENOS
# ================================================================

EPS_ZERO = 1e-5


# ================================================================
# SISTEMA DE EDOs
# ================================================================

def coupled_model(y_vec, t, p):

    (
        V,
        Ap,
        Apm,
        Thn,
        The,
        Tkn,
        Tke,
        B,
        Ps,
        Pl,
        Bm,
        A,
        w,
        x,
        y_inf
    ) = y_vec

    # ------------------------------------------------------------
    # Proteção contra avaliações negativas do integrador
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Subsistema imune
    # ------------------------------------------------------------

    # Calculada apenas para manter a estrutura do modelo.
    # NÃO participa da dinâmica retornada.
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
        + p['πbm1'] * Bmp
        * (1.0 - Bmp / p['πbm2'])
        - p['γbm'] * Bmp
    )

    dA = (
        p['πps'] * Psp
        + p['πpl'] * Plp
        - p['δa'] * Apb
    )

    # ------------------------------------------------------------
    # Subsistema acoplado
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Dinâmica acoplada do vírus
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Condição de contorno não-negativa
    # ------------------------------------------------------------

    y_original = np.asarray(y_vec)

    mask = (
        (y_original <= 0.0)
        &
        (dydt < 0.0)
    )

    dydt[mask] = 0.0

    return dydt


# ================================================================
# SISTEMA NORMALIZADO PARA O ODEINT
# ================================================================

def scaled_model(z, t, p):

    # Retorna ao espaço físico
    y = z * scales

    # Calcula derivadas no espaço físico
    dy = coupled_model(y, t, p)

    # Retorna ao espaço normalizado
    dz = np.asarray(dy, dtype=float) / scales

    return dz


# ================================================================
# FUNÇÃO DE SIMULAÇÃO
# ================================================================

def simulate(p, t):

    # Estado inicial normalizado
    z0 = y0 / scales

    try:

        sol_z, info = odeint(
            scaled_model,
            z0,
            t,
            args=(p,),
            rtol=1e-8,
            atol=1e-10,
            hmax=0.05,
            mxstep=10000,
            full_output=True
        )

    except Exception:
        return None

    # Verifica se o integrador declarou sucesso
    message = info.get('message', '')

    if message != 'Integration successful.':
        return None

    # Verifica valores finitos
    if not np.all(np.isfinite(sol_z)):
        return None

    # Volta para unidades físicas
    sol = sol_z * scales

    # ------------------------------------------------------------
    # Regra solicitada:
    #
    # valores menores que 1e-5 -> zero
    #
    # Primeiro garante não-negatividade e depois aplica o limiar.
    # ------------------------------------------------------------

    sol = np.maximum(sol, 0.0)
    sol[sol < EPS_ZERO] = 0.0

    return sol


# ================================================================
# WRAPPER PARA SOBOL
# ================================================================

def coupled_model_wrapper(X):

    X = np.atleast_1d(X).astype(float).flatten()

    # ------------------------------------------------------------
    # Verificação da quantidade de parâmetros
    # ------------------------------------------------------------

    if len(X) != len(param_keys):
        raise ValueError(
            f"Quantidade incorreta de parâmetros: "
            f"recebidos {len(X)}, "
            f"esperados {len(param_keys)}."
        )

    # ------------------------------------------------------------
    # Cria cópia dos parâmetros nominais
    # ------------------------------------------------------------

    p = params.copy()

    # Substitui pelos valores da amostra Sobol
    for i, key in enumerate(param_keys):
        p[key] = X[i]

    # ------------------------------------------------------------
    # Tempo do Sobol
    # ------------------------------------------------------------

    t = np.linspace(
        0.0,
        500.0,
        N_TIME
    )

    try:

        sol = simulate(p, t)

        # Falha numérica
        if sol is None:
            return np.zeros(
                N_TIME,
                dtype=np.float64
            )

        # --------------------------------------------------------
        # Coluna 11 = Anticorpos A
        # --------------------------------------------------------

        A = sol[:, 11]

        # Verificação final
        if not np.all(np.isfinite(A)):
            return np.zeros(
                N_TIME,
                dtype=np.float64
            )

        # Garantia de shape
        if A.shape != (N_TIME,):
            raise RuntimeError(
                f"Shape inesperado do wrapper: {A.shape}. "
                f"Esperado: {(N_TIME,)}."
            )

        return A.astype(np.float64)

    except Exception:
        return np.zeros(
            N_TIME,
            dtype=np.float64
        )


# ================================================================
# SIMULAÇÃO NOMINAL
# ================================================================

if __name__ == "__main__":

    outdir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "modelo_acoplado_novo_figures"
    )

    os.makedirs(
        outdir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Simulação standalone
    # ------------------------------------------------------------

    t = np.linspace(
        0.0,
        500.0,
        50000
    )

    sol = simulate(
        params,
        t
    )

    if sol is None:
        raise RuntimeError(
            "A simulação nominal falhou."
        )

    (
        V,
        Ap,
        Apm,
        Thn,
        The,
        Tkn,
        Tke,
        B,
        Ps,
        Pl,
        Bm,
        A,
        w,
        x,
        y_inf
    ) = sol.T

    # ------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 10)
    )

    axes[0, 0].plot(
        t,
        V,
        color='purple'
    )
    axes[0, 0].set_title(
        "Vírus / Patógeno (V)"
    )
    axes[0, 0].set_ylabel(
        "Carga Viral"
    )

    axes[0, 1].plot(
        t,
        w,
        color='blue'
    )
    axes[0, 1].set_title(
        "Imunidade Inata (w)"
    )
    axes[0, 1].set_ylabel(
        "Nível de Ativação"
    )

    axes[1, 0].plot(
        t,
        x,
        color='red'
    )
    axes[1, 0].set_title(
        "Dano Tecidual (x)"
    )
    axes[1, 0].set_ylabel(
        "Dano"
    )

    axes[1, 1].plot(
        t,
        y_inf,
        color='orange'
    )
    axes[1, 1].set_title(
        "Inflamação (y)"
    )
    axes[1, 1].set_ylabel(
        "Citocinas"
    )

    axes[2, 0].plot(
        t,
        A,
        color='black'
    )
    axes[2, 0].set_title(
        "Anticorpos (A)"
    )
    axes[2, 0].set_ylabel(
        "Título"
    )

    axes[2, 1].plot(
        t,
        Ap,
        color='brown'
    )
    axes[2, 1].set_title(
        "Células apresentadoras (Ap)"
    )
    axes[2, 1].set_ylabel(
        "Quantidade"
    )

    for ax in axes.flat:
        ax.set_xlabel("Tempo (dias)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    filename = os.path.join(
        outdir,
        "Matsuo_Oscillation_Simulation.png"
    )

    fig.savefig(
        filename,
        dpi=200
    )

    plt.close(fig)

    print(
        f"[OK] Figura salva em: {filename}"
    )

    # ------------------------------------------------------------
    # Teste do wrapper com os valores nominais
    # ------------------------------------------------------------

    X_nom = np.array(
        [params[k] for k in param_keys],
        dtype=float
    )

    qoi = coupled_model_wrapper(
        X_nom
    )

    print(
        "\n[OK] Teste do wrapper"
    )

    print(
        f"     Número de parâmetros = {len(X_nom)}"
    )

    print(
        f"     Número de tempos     = {len(qoi)}"
    )

    print(
        f"     Shape                 = {qoi.shape}"
    )

    print(
        f"     Pico de A             = {np.max(qoi):.6e}"
    )

    print(
        f"     Mínimo de A           = {np.min(qoi):.6e}"
    )