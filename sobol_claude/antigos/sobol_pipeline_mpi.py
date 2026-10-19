"""
sobol_pipeline_mpi.py
─────────────────────
Pipeline MPI de Análise de Sensibilidade de Sobol para o modelo
imunológico acoplado (modelo_acoplado_novo.py).

Arquitetura MPI:
  • Rank 0 gera todas as amostras (SALib) e faz scatter
  • Todos os ranks avaliam sua fatia do modelo em paralelo
  • Rank 0 recolhe (gather) os resultados e calcula os índices de Sobol
  • A decisão de convergência é transmitida a todos com bcast + Barrier

Distribuição dos parâmetros:
  Lognormal(s=0.25, scale=0.1*nominal) — idêntica ao uq_sa.ipynb de referência.
  No SALib: dists='lognorm', bounds=[log(0.1*nominal), 0.25] por parâmetro.

Fluxo por rodada (N = número de amostras base de Sobol):
  Total de avaliações de modelo = N * (num_vars + 2) = N * 12

Sequência de convergência:
  100k → 110k → (se convergido) → confirma com 90k

Uso:
  mpiexec -n 32 python sobol_pipeline_mpi.py
  mpiexec -n 32 python sobol_pipeline_mpi.py --outdir ./resultados --tol 0.05

ATENÇÃO: modelo_acoplado_novo.py deve estar no mesmo diretório.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from mpi4py import MPI
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer

# ─────────────────────────────────────────────────────────────────
# Inicialização MPI
# ─────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ─────────────────────────────────────────────────────────────────
# Configuração do problema de Sobol
#
# CRÍTICO: 'names' deve ser idêntico (mesma ordem) a
# param_keys em modelo_acoplado_novo.py.
#
# Distribuição: Lognormal(s=0.25, scale=0.1*nominal)
# No SALib com dists='lognorm': bounds=[log(scale), sigma]
#   → bounds=[log(0.1*nominal), 0.25]
# ─────────────────────────────────────────────────────────────────
_S  = 0.25   # sigma da log-normal
_SF = 0.1    # scale = SF * nominal (como no uq_sa.ipynb de referência)

_NOMINAL = {
    'αtk': 2.17e-4,
    'βtk': 1e-5,
    'πps': 2e-3,
    'πtk': 1e-8,
    'δps': 2.0e0,
    'αb':  6.0e0,
    'H1':  1e-4,
    'f':   2.0,
    'c':   0.2,
    'r':   1.0,
}

PARAM_NAMES = list(_NOMINAL.keys())   # ordem canônica — deve bater com param_keys do modelo

LATEX_LABELS = {
    'αtk': r'$\alpha_{tk}$',
    'βtk': r'$\beta_{tk}$',
    'πps': r'$\pi_{ps}$',
    'πtk': r'$\pi_{tk}$',
    'δps': r'$\delta_{ps}$',
    'αb':  r'$\alpha_b$',
    'H1':  r'$H_1$',
    'f':   r'$f$',
    'c':   r'$c$',
    'r':   r'$r$',
}

problem = {
    'num_vars': len(PARAM_NAMES),
    'names':    PARAM_NAMES,
    'bounds':   [[np.log(_SF * _NOMINAL[k]), _S] for k in PARAM_NAMES],
    'dists':    ['lognorm'] * len(PARAM_NAMES),
}

# ─────────────────────────────────────────────────────────────────
# Importar o modelo (todos os ranks precisam da função)
# ─────────────────────────────────────────────────────────────────
from modelo_acoplado_novo import coupled_wrapper


# ─────────────────────────────────────────────────────────────────
# Funções auxiliares (rank 0)
# ─────────────────────────────────────────────────────────────────

def _generate_samples(N: int, seed: int = 42) -> np.ndarray:
    """Gera N*(num_vars+2) amostras via SALib Sobol/Saltelli."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suprime aviso de potência de 2
        X = sobol_sampler.sample(
            problem,
            N,
            calc_second_order=False,
            seed=seed,
        )
    return X   # shape: (N*(num_vars+2), num_vars)


def _save_samples(X: np.ndarray, outdir: str, N: int):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(X, columns=PARAM_NAMES)
    df.to_csv(os.path.join(outdir, f"samples_{N}.csv"), index=False)


def _run_sobol_analysis(Y: np.ndarray) -> dict:
    return sobol_analyzer.analyze(
        problem,
        Y,
        calc_second_order=False,
        num_resamples=1000,
        conf_level=0.95,
        print_to_console=False,
    )


def _save_results(Si: dict, Y: np.ndarray, outdir: str, N: int) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)

    df = pd.DataFrame({
        "param":    PARAM_NAMES,
        "latex":    [LATEX_LABELS[k] for k in PARAM_NAMES],
        "S1":       Si["S1"],
        "S1_conf":  Si["S1_conf"],
        "ST":       Si["ST"],
        "ST_conf":  Si["ST_conf"],
    })
    df.to_csv(os.path.join(outdir, f"sobol_{N}.csv"), index=False)
    np.save(os.path.join(outdir, f"qoi_{N}.npy"), Y)

    return df


def _print_table(df: pd.DataFrame, N: int):
    print(f"\n  {'Parâmetro':<8} {'S1':>10} {'+-S1':>8} {'ST':>10} {'±ST':>8}")
    print("  " + "-" * 48)
    for _, row in df.iterrows():
            print(
                f"  {row['param']:<8} "
                f"{row['S1']:>10.5f} "
                f"{row['S1_conf']:>8.5f} "
                f"{row['ST']:>10.5f} "
                f"{row['ST_conf']:>8.5f}"
                )


def _check_convergence(si_a: dict, si_b: dict, tol: float,
                       na: int, nb: int) -> bool:
    """
    Diferença relativa máxima nos S1, considerando apenas
    parâmetros com |S1| > 1e-3 (evita divisão por zero).
    """
    s1_a = np.array(si_a["S1"])
    s1_b = np.array(si_b["S1"])

    mask = np.abs(s1_a) > 1e-3
    if mask.sum() == 0:
        print(f"\n  [Convergência] {na:,} → {nb:,}: todos S1 ≈ 0 — CONVERGIDO")
        return True

    rel_diff = np.abs(s1_b[mask] - s1_a[mask]) / np.abs(s1_a[mask])
    max_diff = float(np.max(rel_diff))

    converged = max_diff < tol
    print(f"\n  [Convergência] {na:,} → {nb:,}")
    print(f"  Diferença relativa máxima em S1: {max_diff:.4f}  (tolerância: {tol})")
    print(f"  {'✓ CONVERGIDO' if converged else '✗ não convergiu'}")
    return converged


# ─────────────────────────────────────────────────────────────────
# Núcleo: uma rodada completa
# ─────────────────────────────────────────────────────────────────

def execute_case(N: int, outdir: str, seed: int = 42) -> dict | None:
    """
    Executa uma rodada Sobol completa com N amostras base.
    Todos os ranks devem chamar esta função.
    Retorna o dicionário Si do SALib somente no rank 0; None nos demais.
    """

    # ── 1. Rank 0 gera as amostras e faz scatter ────────────────
    if rank == 0:
        t0 = time.time()
        print(f"\n  [1/3] Gerando amostras  (N={N:,})...", flush=True)
        X = _generate_samples(N, seed=seed)
        print(f"        Total de avaliações: {X.shape[0]:,}", flush=True)
        _save_samples(X, outdir, N)
        chunks = np.array_split(X, size)
    else:
        chunks = None

    local_X = comm.scatter(chunks, root=0)

    # ── 2. Todos os ranks avaliam o modelo ──────────────────────
    if rank == 0:
        print(f"\n  [2/3] Avaliando modelo em {size} processo(s) MPI...", flush=True)

    n_local = len(local_X)
    local_Y = np.empty(n_local, dtype=np.float64)

    t_local = time.time()
    log_step = max(1, n_local // 10)

    for i, row in enumerate(local_X):
        local_Y[i] = coupled_wrapper(row)

        if rank == 0 and (i + 1) % log_step == 0:
            elapsed = time.time() - t_local
            eta = (n_local - i - 1) / ((i + 1) / elapsed)
            print(
                f"        rank 0: {i+1:>7}/{n_local}  "
                f"({elapsed/60:.1f} min decorrido, ETA {eta/60:.1f} min)",
                flush=True,
            )

    # ── 3. Gather e análise (rank 0) ────────────────────────────
    gathered = comm.gather(local_Y, root=0)

    if rank == 0:
        Y = np.concatenate(gathered)

        elapsed_total = time.time() - t0
        print(f"\n  [3/3] Análise de Sobol...", flush=True)
        print(f"        Tempo de avaliação: {elapsed_total/60:.1f} min")
        print(
            f"        Y — min={Y.min():.4e}  max={Y.max():.4e}  "
            f"NaN={np.isnan(Y).sum()}  Inf={np.isinf(Y).sum()}"
        )

        bad = ~np.isfinite(Y)
        if bad.sum() > 0:
            print(f"  [AVISO] {bad.sum()} valores inválidos substituídos por 0")
            Y[bad] = 0.0

        Si = _run_sobol_analysis(Y)
        df = _save_results(Si, Y, outdir, N)
        _print_table(df, N)

        return Si

    return None


# ─────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline MPI Sobol — modelo imunológico acoplado"
    )
    parser.add_argument(
        "--outdir", type=str, default="./resultados_sobol",
        help="Diretório de saída (default: ./resultados_sobol)"
    )
    parser.add_argument(
        "--tol", type=float, default=0.05,
        help="Tolerância de convergência nos S1 (default: 0.05)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente aleatória para o SALib (default: 42)"
    )
    args = parser.parse_args()

    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"\n{'#'*60}")
        print(f"# Pipeline Sobol MPI — modelo acoplado — {size} processo(s)")
        print(f"# outdir : {args.outdir}")
        print(f"# tol    : {args.tol}")
        print(f"# seed   : {args.seed}")
        print(f"{'#'*60}")

    # ── Rodada 1: 100k ──────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}\n  RODADA 1 — N = 100 000\n{'='*60}")
    si_100k = execute_case(100_000, args.outdir, seed=args.seed)
    comm.Barrier()

    # ── Rodada 2: 110k ──────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}\n  RODADA 2 — N = 110 000\n{'='*60}")
    si_110k = execute_case(110_000, args.outdir, seed=args.seed)
    comm.Barrier()

    # ── Verificação de convergência ──────────────────────────────
    if rank == 0:
        converged = _check_convergence(si_100k, si_110k, args.tol, 100_000, 110_000)
    else:
        converged = None
    converged = comm.bcast(converged, root=0)
    comm.Barrier()

    # ── Confirmação com 90k ──────────────────────────────────────
    if converged:
        if rank == 0:
            print(f"\n{'='*60}\n  CONFIRMAÇÃO — N = 90 000\n{'='*60}")
        si_90k = execute_case(90_000, args.outdir, seed=args.seed)
        comm.Barrier()

        if rank == 0:
            conv2 = _check_convergence(si_90k, si_100k, args.tol, 90_000, 100_000)
            if conv2:
                print("\n[RESULTADO FINAL] ✓ Convergência confirmada com 90k.")
                print("  Os resultados com N=100k são confiáveis.")
            else:
                print("\n[RESULTADO FINAL] ✗ 90k não confirmou 100k.")
                print("  Revise a tolerância ou aumente N além de 110k.")
    else:
        if rank == 0:
            print("\n[RESULTADO FINAL] ✗ 100k e 110k não convergiram.")
            print("  Aumente N além de 110k.")

    # ── Resumo final ─────────────────────────────────────────────
    if rank == 0:
        csv_path = os.path.join(args.outdir, "sobol_100000.csv")
        if os.path.exists(csv_path):
            df_final = pd.read_csv(csv_path)
            print(f"\n{'='*60}")
            print("  RESUMO — Top 5 por S1  (rodada N=100k)")
            print(f"{'='*60}")
            for _, row in df_final.nlargest(5, "S1").iterrows():
                print(
                    f"  {row['latex']:<16}  "
                    f"S1 = {row['S1']:.4f} ± {row['S1_conf']:.4f}  |  "
                    f"ST = {row['ST']:.4f} ± {row['ST_conf']:.4f}"
                )
        print(f"\n[DONE] Resultados em: {args.outdir}\n")


if __name__ == "__main__":
    main()
