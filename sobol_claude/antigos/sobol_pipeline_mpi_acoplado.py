"""
sobol_pipeline_mpi.py
---------------------
Análise temporal de sensibilidade de Sobol usando MPI.

Modelo:
    teste_gpt_sobol.py

Parâmetros analisados:
    πv
    kv1
    πps
    kv2
    δps
    αb
    H1
    f
    c
    r

QoI temporal:
    A(t) = Anticorpos

Para cada amostra de parâmetros, o wrapper retorna 100 valores
temporais.

A análise produz:
    S1(t) = índice Sobol de primeira ordem
    ST(t) = índice Sobol de efeito total

Uso:

    mpirun -np 4 python sobol_pipeline_mpi.py

ou:

    mpirun -np 8 python sobol_pipeline_mpi.py --outdir resultados

Exemplo com N menor para teste:

    mpirun -np 4 python sobol_pipeline_mpi.py --N 1024
"""

import argparse
import os
import time
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpi4py import MPI

from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer

from teste_gpt_sobol import coupled_model_wrapper


# ================================================================
# MPI
# ================================================================

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


# ================================================================
# CONFIGURAÇÃO DA DISTRIBUIÇÃO
# ================================================================
#
# Mantida a lógica do seu código original:
#
#   média / posição no log:
#       log(0.1 * nominal)
#
#   desvio logarítmico:
#       0.25
#
# ================================================================

_S = 0.25
_SF = 0.1


# ================================================================
# PARÂMETROS NOMINAIS
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


# ================================================================
# ORDEM DOS PARÂMETROS
# ================================================================

PARAM_NAMES = list(
    _NOMINAL.keys()
)


# ================================================================
# RÓTULOS DOS GRÁFICOS
# ================================================================

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
# PROBLEMA SOBOL
# ================================================================

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

    'dists': [
        'lognorm'
    ] * len(PARAM_NAMES),
}


# ================================================================
# CONFIGURAÇÃO TEMPORAL
# ================================================================

N_TIME = 100

T_START = 0.0
T_END = 500.0

T_VALUES = np.linspace(
    T_START,
    T_END,
    N_TIME
)


# ================================================================
# AMOSTRAGEM SOBOL
# ================================================================

def _generate_samples(
    N: int,
    seed: int = 42
):

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        X = sobol_sampler.sample(
            problem,
            N,
            calc_second_order=False,
            seed=seed
        )

    return X


# ================================================================
# CRIA MATRIZ DE RECEPÇÃO MPI
# ================================================================

def _build_gatherv_info(
    n_total,
    n_ranks,
    n_time
):

    # Distribuição de linhas equivalente ao array_split
    base = n_total // n_ranks
    resto = n_total % n_ranks

    rows_per_rank = [
        base + (1 if r < resto else 0)
        for r in range(n_ranks)
    ]

    counts = [
        n_rows * n_time
        for n_rows in rows_per_rank
    ]

    displacements = np.zeros(
        n_ranks,
        dtype=np.int64
    )

    for i in range(1, n_ranks):
        displacements[i] = (
            displacements[i - 1]
            + counts[i - 1]
        )

    return (
        rows_per_rank,
        np.asarray(counts, dtype=np.int64),
        displacements
    )


# ================================================================
# EXECUÇÃO DO SOBOL TEMPORAL
# ================================================================

def execute_temporal_sobol(
    N: int,
    outdir: str,
    seed: int = 42
):

    # ============================================================
    # ETAPA 1 - GERAÇÃO DAS AMOSTRAS
    # ============================================================

    if rank == 0:

        t_global_start = time.time()

        print(
            f"\n[1/4] Gerando amostras Sobol..."
        )

        print(
            f"      N base       = {N:,}"
        )

        print(
            f"      D             = {len(PARAM_NAMES)}"
        )

        print(
            f"      Segunda ordem = não"
        )

        X = _generate_samples(
            N,
            seed=seed
        )

        n_total = X.shape[0]

        print(
            f"      Avaliações    = {n_total:,}"
        )

        print(
            f"      X.shape       = {X.shape}"
        )

        # Informação de distribuição por rank
        rows_per_rank, counts, displacements = (
            _build_gatherv_info(
                n_total,
                size,
                N_TIME
            )
        )

        # array_split, compatível com a distribuição acima
        chunks = np.array_split(
            X,
            size
        )

    else:

        X = None
        chunks = None

        rows_per_rank = None
        counts = None
        displacements = None


    # ============================================================
    # DISTRIBUIÇÃO MPI
    # ============================================================

    local_X = comm.scatter(
        chunks,
        root=0
    )

    n_local = len(local_X)

    if rank == 0:

        print(
            f"\n[2/4] Resolvendo as EDOs..."
        )

        print(
            f"      Processos MPI = {size}"
        )

        print(
            f"      N_TIME        = {N_TIME}"
        )


    # ============================================================
    # MATRIZ LOCAL
    # ============================================================

    local_Y = np.zeros(
        (
            n_local,
            N_TIME
        ),
        dtype=np.float64
    )


    t_local = time.time()


    # ============================================================
    # EXECUÇÃO DO MODELO
    # ============================================================

    for i, row in enumerate(local_X):

        result = coupled_model_wrapper(
            row
        )

        # --------------------------------------------------------
        # Verificação do wrapper
        # --------------------------------------------------------

        if result.shape != (N_TIME,):

            raise RuntimeError(
                f"[rank {rank}] "
                f"Wrapper retornou shape "
                f"{result.shape}, "
                f"esperado {(N_TIME,)}"
            )

        if not np.all(
            np.isfinite(result)
        ):

            raise RuntimeError(
                f"[rank {rank}] "
                f"Wrapper retornou valores "
                f"não finitos."
            )

        local_Y[i, :] = result


        # --------------------------------------------------------
        # Progresso
        # --------------------------------------------------------

        if rank == 0:

            step = max(
                1,
                n_local // 10
            )

            if (
                (i + 1) % step == 0
                or i + 1 == n_local
            ):

                elapsed = (
                    time.time()
                    - t_local
                )

                print(
                    f"      Progresso local: "
                    f"{i + 1:>7}/{n_local} "
                    f"({100 * (i + 1) / n_local:5.1f}%) "
                    f"[{elapsed / 60:.1f} min]",
                    flush=True
                )


    # ============================================================
    # BARREIRA
    # ============================================================

    comm.Barrier()


    # ============================================================
    # ETAPA 3 - REUNIÃO DOS RESULTADOS
    # ============================================================

    if rank == 0:

        print(
            "\n[3/4] Reunindo resultados..."
        )

        # --------------------------------------------------------
        # Diretório
        # --------------------------------------------------------

        os.makedirs(
            outdir,
            exist_ok=True
        )

        # --------------------------------------------------------
        # Memmap para evitar duas grandes cópias na RAM
        # --------------------------------------------------------

        y_file = os.path.join(
            outdir,
            f"Y_temporal_{N}.dat"
        )

        Y = np.memmap(
            y_file,
            dtype=np.float64,
            mode='w+',
            shape=(
                n_total,
                N_TIME
            )
        )

    else:

        Y = None


    # ============================================================
    # GATHERV
    # ============================================================

    sendbuf = local_Y.ravel()


    if rank == 0:

        comm.Gatherv(

            sendbuf,

            [
                Y.ravel(),
                counts.tolist(),
                displacements.tolist(),
                MPI.DOUBLE
            ],

            root=0
        )

        Y.flush()

    else:

        comm.Gatherv(
            sendbuf,
            None,
            root=0
        )


    # ============================================================
    # ANÁLISE DE SOBOL TEMPORAL
    # ============================================================

    if rank == 0:

        print(
            "\n[4/4] Calculando índices de Sobol..."
        )

        print(
            f"      Matriz Y = {Y.shape}"
        )

        print(
            f"      Arquivo   = {y_file}"
        )

        ST_temporal = np.zeros(
            (
                N_TIME,
                len(PARAM_NAMES)
            ),
            dtype=float
        )

        S1_temporal = np.zeros(
            (
                N_TIME,
                len(PARAM_NAMES)
            ),
            dtype=float
        )

        # --------------------------------------------------------
        # Loop temporal
        # --------------------------------------------------------

        for t_idx in range(N_TIME):

            # ----------------------------------------------------
            # Copia de uma única coluna para RAM
            #
            # Isso evita trazer a matriz inteira da memmap para
            # memória a cada análise.
            # ----------------------------------------------------

            Y_t = np.asarray(
                Y[:, t_idx]
            ).copy()

            # ----------------------------------------------------
            # Análise Sobol
            # ----------------------------------------------------

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                Si = sobol_analyzer.analyze(

                    problem,
                    Y_t,

                    calc_second_order=False,

                    num_resamples=100,

                    print_to_console=False
                )


            S1_temporal[t_idx, :] = (
                Si["S1"]
            )

            ST_temporal[t_idx, :] = (
                Si["ST"]
            )


            # ----------------------------------------------------
            # Progresso
            # ----------------------------------------------------

            step = max(
                1,
                N_TIME // 10
            )

            if (
                (t_idx + 1) % step == 0
                or t_idx + 1 == N_TIME
            ):

                print(
                    f"      Tempo: "
                    f"{t_idx + 1:>3}/{N_TIME} "
                    f"({100 * (t_idx + 1) / N_TIME:5.1f}%)",
                    flush=True
                )


        # ========================================================
        # SALVA RESULTADOS NUMÉRICOS
        # ========================================================

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


        # ========================================================
        # GRÁFICO ST
        # ========================================================

        plt.figure(
            figsize=(9, 6),
            dpi=300
        )

        for i, p_name in enumerate(
            PARAM_NAMES
        ):

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


        # ========================================================
        # GRÁFICO S1
        # ========================================================

        plt.figure(
            figsize=(9, 6),
            dpi=300
        )

        for i, p_name in enumerate(
            PARAM_NAMES
        ):

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


        # ========================================================
        # REMOVE MAPA TEMPORÁRIO
        # ========================================================
        #
        # O arquivo .dat contém a matriz completa Y.
        # Caso queira mantê-lo para análise posterior, basta
        # comentar o os.remove().
        #

        try:
            del Y
            os.remove(y_file)

        except Exception:
            pass


        # ========================================================
        # RELATÓRIO FINAL
        # ========================================================

        total_elapsed = (
            time.time()
            - t_global_start
        )

        print(
            "\n============================================"
        )

        print(
            "ANÁLISE SOBOL FINALIZADA"
        )

        print(
            "============================================"
        )

        print(
            f"N base              : {N:,}"
        )

        print(
            f"N parâmetros        : {len(PARAM_NAMES)}"
        )

        print(
            f"N avaliações        : {n_total:,}"
        )

        print(
            f"N pontos temporais  : {N_TIME}"
        )

        print(
            f"Tempo total         : "
            f"{total_elapsed / 3600:.2f} h"
        )

        print(
            f"\nResultados salvos em:"
        )

        print(
            f"  {outdir}"
        )

        print(
            f"\nArquivos:"
        )

        print(
            f"  sobol_temporal_S1_{N}.png"
        )

        print(
            f"  sobol_temporal_ST_{N}.png"
        )

        print(
            f"  S1_temporal_{N}.csv"
        )

        print(
            f"  ST_temporal_{N}.csv"
        )

        print(
            "============================================"
        )

    # ============================================================
    # GARANTE FINALIZAÇÃO DE TODOS OS RANKS
    # ============================================================

    comm.Barrier()


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Análise temporal de Sobol "
            "com MPI para o modelo imunológico acoplado."
        )
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./resultados_acoplado",
        help="Diretório base dos resultados."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed da amostragem Sobol."
    )

    args = parser.parse_args()

    # ============================================================
    # TRÊS VALORES DE N PARA O TESTE
    # ============================================================
    #
    # 65536  = 2^16
    # 98304  = 3 * 2^15
    # 131072 = 2^17
    #
    # Esses valores ficam próximos da faixa recomendada:
    # 90000, 100000 e 110000.
    # ============================================================

    N_VALUES = [
        65_536,
        98_304,
        131_072
    ]

    for run_id, N in enumerate(N_VALUES, start=1):

        if rank == 0:

            print()
            print("=" * 70)
            print(
                f"EXECUÇÃO {run_id}/3"
            )
            print(
                f"N base = {N:,}"
            )
            print("=" * 70)

        # --------------------------------------------------------
        # Cada N terá seu próprio diretório
        # --------------------------------------------------------

        run_outdir = os.path.join(
            args.outdir,
            f"N_{N}"
        )

        # --------------------------------------------------------
        # Executa análise completa
        # --------------------------------------------------------

        execute_temporal_sobol(
            N=N,
            outdir=run_outdir,
            seed=args.seed
        )

        # --------------------------------------------------------
        # Garante que todos os processos terminaram esta execução
        # antes de iniciar a próxima.
        # --------------------------------------------------------

        comm.Barrier()

        if rank == 0:

            print()
            print(
                f"[OK] Execução {run_id}/3 concluída para N={N:,}"
            )
            print()


if __name__ == "__main__":
    main()