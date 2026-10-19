import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SALib.analyze import sobol as sobol_analyzer


# ================================================================
# CONFIGURAÇÃO
# ================================================================

N = 1024

outdir = "./resultados_acoplado"

N_TIME = 100

T_VALUES = np.linspace(
    0.0,
    500.0,
    N_TIME
)

PARAM_NAMES = [
    'πv',
    'kv1',
    'πps',
    'kv2',
    'δps',
    'αb',
    'H1',
    'f',
    'c',
    'r',
]

LATEX_LABELS = {
    'πv':  r'$\pi_v$',
    'kv1': r'$k_{v1}$',
    'πps': r'$\pi_{ps}$',
    'kv2': r'$k_{v2}$',
    'δps': r'$\delta_{ps}$',
    'αb':  r'$\alpha_b$',
    'H1':  r'$H_1$',
    'f':   r'$f$',
    'c':   r'$c$',
    'r':   r'$r$',
}


# ================================================================
# RECONSTRÓI O MESMO PROBLEM DO SALIB
# ================================================================

_NOMINAL = {
    'πv':  6.80e-1,
    'kv1': 4.82e-5,
    'πps': 2e-3,
    'kv2': 7.48e-7,
    'δps': 2.0e0,
    'αb':  6.0e0,
    'H1':  1e-4,
    'f':   2.0,
    'c':   0.2,
    'r':   1.0,
}

_S = 0.25
_SF = 0.1

problem = {
    'num_vars': len(PARAM_NAMES),
    'names': PARAM_NAMES,
    'bounds': [
        [
            np.log(_SF * _NOMINAL[k]),
            _S
        ]
        for k in PARAM_NAMES
    ],
    'dists': ['lognorm'] * len(PARAM_NAMES),
}


# ================================================================
# LOCALIZA O Y
# ================================================================

y_file = os.path.join(
    outdir,
    f"Y_temporal_{N}.dat"
)

if not os.path.exists(y_file):
    raise FileNotFoundError(
        f"Arquivo não encontrado:\n{y_file}"
    )


# Para N=1024, D=10 e calc_second_order=False:
#
# avaliações = N * (D + 2)
#
n_total = N * (len(PARAM_NAMES) + 2)

print("Carregando Y...")
print(f"Arquivo: {y_file}")
print(f"Shape esperado: ({n_total}, {N_TIME})")


Y = np.memmap(
    y_file,
    dtype=np.float64,
    mode="r",
    shape=(n_total, N_TIME)
)

print(f"Y.shape = {Y.shape}")


# ================================================================
# SOBOL TEMPORAL
# ================================================================

S1_temporal = np.zeros(
    (N_TIME, len(PARAM_NAMES))
)

ST_temporal = np.zeros(
    (N_TIME, len(PARAM_NAMES))
)


for t_idx in range(N_TIME):

    Y_t = np.asarray(
        Y[:, t_idx]
    ).copy()

    Si = sobol_analyzer.analyze(
        problem,
        Y_t,
        calc_second_order=False,
        num_resamples=100,
        print_to_console=False
    )

    S1_temporal[t_idx, :] = Si["S1"]
    ST_temporal[t_idx, :] = Si["ST"]

    if (
        (t_idx + 1) % 10 == 0
        or t_idx + 1 == N_TIME
    ):
        print(
            f"Tempo: {t_idx + 1:3d}/{N_TIME}"
        )


# ================================================================
# SALVA CSV EM UTF-8
# ================================================================

os.makedirs(
    outdir,
    exist_ok=True
)


np.savetxt(
    os.path.join(
        outdir,
        f"S1_temporal_{N}.csv"
    ),
    np.column_stack(
        [
            T_VALUES,
            S1_temporal
        ]
    ),
    delimiter=",",
    header="time," + ",".join(PARAM_NAMES),
    comments="",
    encoding="utf-8"
)


np.savetxt(
    os.path.join(
        outdir,
        f"ST_temporal_{N}.csv"
    ),
    np.column_stack(
        [
            T_VALUES,
            ST_temporal
        ]
    ),
    delimiter=",",
    header="time," + ",".join(PARAM_NAMES),
    comments="",
    encoding="utf-8"
)


# ================================================================
# GRÁFICO ST
# ================================================================

plt.figure(
    figsize=(9, 6),
    dpi=300
)

for i, p_name in enumerate(PARAM_NAMES):

    plt.plot(
        T_VALUES,
        ST_temporal[:, i],
        label=LATEX_LABELS[p_name],
        linewidth=2
    )

plt.title(
    "Sobol Total (ST) ao Longo do Tempo"
)

plt.xlabel(
    "Tempo (dias)"
)

plt.ylabel(
    "Índice ST"
)

plt.grid(
    True
)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        outdir,
        f"sobol_temporal_ST_{N}.png"
    )
)

plt.close()


# ================================================================
# GRÁFICO S1
# ================================================================

plt.figure(
    figsize=(9, 6),
    dpi=300
)

for i, p_name in enumerate(PARAM_NAMES):

    plt.plot(
        T_VALUES,
        S1_temporal[:, i],
        label=LATEX_LABELS[p_name],
        linewidth=2
    )

plt.title(
    "Sobol Primeira Ordem (S1) ao Longo do Tempo"
)

plt.xlabel(
    "Tempo (dias)"
)

plt.ylabel(
    "Índice S1"
)

plt.grid(
    True
)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        outdir,
        f"sobol_temporal_S1_{N}.png"
    )
)

plt.close()


print()
print("============================================")
print("RECUPERAÇÃO FINALIZADA")
print("============================================")
print(f"Resultados salvos em: {outdir}")
print()
print(f"S1_temporal_{N}.csv")
print(f"ST_temporal_{N}.csv")
print(f"sobol_temporal_S1_{N}.png")
print(f"sobol_temporal_ST_{N}.png")
print("============================================")