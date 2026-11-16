import argparse
import csv
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from scipy.integrate import solve_ivp
from uqpylab import sessions

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if __name__ == "__main__":
    sys.modules["sobol_uqpylab"] = sys.modules[__name__]

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

# 3. Definição do problema (mesmas variáveis e limites do SALib)
problem = {
    'num_vars': 10,
    'names': ['H1', 'H2', 'H3', 'B1', 'B2', 'B3', 'c', 'g', 'r', 'πps'],
    'bounds': [
        [0.5e-4, 1.5e-4],  # Limites para H1
        [0.5e-9, 1.5e-9],  # Limites para H2
        [0.5e-10, 1.5e-10],  # Limites para H3
        [0.5e-4, 1.5e-4],  # Limites para B1
        [0.5e-9, 1.5e-9],  # Limites para B2
        [0.5e-10, 1.5e-10],  # Limites para B3
        [0.1, 0.3],        # Limites para c
        [0.1, 0.3],        # Limites para g
        [0.5, 1.5],        # Limites para r
        [1e-3, 3e-3],      # Limites para πps
    ]
}

# 4. Avaliação de uma amostra
y0 = [1.0, 1e6, 0, 1e6, 0, 5e5, 0, 2.5e5, 0, 0, 0, 150, 0.1, 0.0, 0.0]
t_span = (0, 500)
time_points = np.linspace(t_span[0], t_span[1], 101)

_WORKER_STOP = 0
_ACTIVE_OUTPUT_INDEX = None
_OUTPUT_CACHE_INDEX = {}
_OUTPUT_CACHE_VALUES = []
_progress = {
    "completed": 0,
    "total": None,
    "start": None,
    "N": None,
}


def evaluate_sample(X):
    p = base_params.copy()
    for name, value in zip(problem['names'], X):
        p[name] = value

    sol = solve_ivp(
        coupled_model_ivp,
        t_span,
        y0,
        args=(p,),
        method='LSODA',
        max_step=1.0,
        t_eval=time_points,
    )
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError(sol.message)

    return sol.y[13, :].copy()


def _evaluate_output_matrix(samples, output_index):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return np.empty((0, 1), dtype=np.float64), 0
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    outputs = np.empty((samples.shape[0], 1), dtype=np.float64)
    new_evaluations = 0
    for i, sample in enumerate(samples):
        key = sample.tobytes()
        cached_index = _OUTPUT_CACHE_INDEX.get(key)
        if cached_index is None:
            trajectory = evaluate_sample(sample)[1:]
            cached_index = len(_OUTPUT_CACHE_VALUES)
            _OUTPUT_CACHE_INDEX[key] = cached_index
            _OUTPUT_CACHE_VALUES.append(trajectory)
            new_evaluations += 1
        outputs[i, 0] = _OUTPUT_CACHE_VALUES[cached_index][output_index]
    return outputs, new_evaluations


def _report_progress(batch_size):
    if _progress["start"] is None:
        return
    _progress["completed"] += int(batch_size)
    elapsed = time.perf_counter() - _progress["start"]
    if _progress["total"]:
        progress = 100.0 * _progress["completed"] / _progress["total"]
        print(
            f"\rN={_progress['N']:,} | progresso: {progress:6.2f}% "
            f"({_progress['completed']:,}/{_progress['total']:,}) "
            f"| decorrido: {elapsed / 60:.1f} min",
            end="",
            flush=True,
        )
    else:
        print(
            f"\rN={_progress['N']:,} | simulações: {_progress['completed']:,} "
            f"| decorrido: {elapsed / 60:.1f} min",
            end="",
            flush=True,
        )


def evaluate_samples(X, parameters=None):
    """Função vetorizada chamada pelo UQPyLab durante a análise de Sobol."""
    del parameters
    if _ACTIVE_OUTPUT_INDEX is None:
        raise RuntimeError("Nenhum instante temporal foi selecionado para a análise.")
    output_index = _ACTIVE_OUTPUT_INDEX

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if size == 1:
        outputs, new_evaluations = _evaluate_output_matrix(X, output_index)
        _report_progress(new_evaluations)
        return outputs[:, 0]

    if rank != 0:
        raise RuntimeError("evaluate_samples deve ser chamada apenas no rank 0.")

    if not 0 <= output_index < len(time_points) - 1:
        raise RuntimeError(f"Índice de saída temporal inválido: {output_index}.")
    comm.bcast((int(X.shape[0]), output_index), root=0)
    chunks = np.array_split(X, size)
    local_samples = comm.scatter(chunks, root=0)
    local_outputs, local_new_evaluations = _evaluate_output_matrix(
        local_samples, output_index
    )
    gathered = comm.gather(local_outputs, root=0)
    new_evaluations = comm.gather(local_new_evaluations, root=0)
    outputs = np.vstack(gathered)
    _report_progress(sum(new_evaluations))
    return outputs[:, 0]


def _worker_loop():
    comm = MPI.COMM_WORLD
    while True:
        work = comm.bcast(None, root=0)
        if work == _WORKER_STOP:
            break
        n_samples, output_index = work
        local_samples = comm.scatter(None, root=0)
        local_outputs, local_new_evaluations = _evaluate_output_matrix(
            local_samples, output_index
        )
        comm.gather(local_outputs, root=0)
        comm.gather(local_new_evaluations, root=0)


def _stop_workers():
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1 and comm.Get_rank() == 0:
        comm.bcast(_WORKER_STOP, root=0)


def save_results(output_dir, indices):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = problem['names']
    s1 = indices['S1']
    st = indices['ST']
    s1_conf = indices['S1_conf']
    st_conf = indices['ST_conf']

    csv_path = output_dir / "indices_sobol.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["tempo_dias"]
        for name in names:
            header.extend((f"S1_{name}", f"S1_conf_{name}", f"ST_{name}", f"ST_conf_{name}"))
        writer.writerow(header)
        for time_index, time_value in enumerate(time_points):
            row = [time_value]
            for parameter_index in range(len(names)):
                row.extend((
                    s1[parameter_index, time_index],
                    s1_conf[parameter_index, time_index],
                    st[parameter_index, time_index],
                    st_conf[parameter_index, time_index],
                ))
            writer.writerow(row)

    for index_name, values, errors, color in (
        ("S1", s1, s1_conf, "tab:blue"),
        ("ST", st, st_conf, "tab:orange"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for parameter_index, name in enumerate(names):
            line = ax.plot(
                time_points,
                values[parameter_index],
                label=name,
                color=f"C{parameter_index}",
            )[0]
            ax.fill_between(
                time_points,
                values[parameter_index] - errors[parameter_index],
                values[parameter_index] + errors[parameter_index],
                color=line.get_color(),
                alpha=0.15,
            )
        ax.set_xlabel("Tempo (dias)")
        ax.set_ylabel(f"Índice {index_name}")
        ax.set_title(f"Evolução temporal dos índices {index_name} de Sobol")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(title="Parâmetro")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{index_name}_tempo.png", dpi=200)
        plt.close(fig)

    for i, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(5, 4))
        for index_name, values, errors, color in (
            ("S1", s1, s1_conf, "tab:blue"),
            ("ST", st, st_conf, "tab:orange"),
        ):
            ax.plot(time_points, values[i], label=index_name, color=color)
            ax.fill_between(
                time_points,
                values[i] - errors[i],
                values[i] + errors[i],
                color=color,
                alpha=0.15,
            )
        ax.set_xlabel("Tempo (dias)")
        ax.set_ylabel("Índice de sensibilidade")
        ax.set_title(f"Evolução temporal dos índices de {name}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"sobol_temporal_{name}.png", dpi=200)
        plt.close(fig)


def _as_var_by_output(values, n_vars, n_outputs):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(n_vars, 1)
    if array.shape[0] == n_vars and array.shape[1] == n_outputs:
        return array
    if array.shape[0] == n_outputs and array.shape[1] == n_vars:
        return array.T
    raise RuntimeError(
        f"Formato inesperado dos índices de Sobol: {array.shape} "
        f"(esperado ({n_vars}, {n_outputs}))."
    )


def _first_order_bootstrap(bootstrap):
    if not bootstrap:
        return {}
    all_orders = bootstrap.get("AllOrders")
    if isinstance(all_orders, dict):
        return all_orders
    if isinstance(all_orders, list) and all_orders:
        return all_orders[0]
    return bootstrap.get("FirstOrder") or {}


def _bootstrap_halfwidth(bootstrap_block, estimate):
    if not bootstrap_block:
        return np.zeros_like(estimate)
    ci = np.asarray(bootstrap_block.get("CI", []), dtype=float)
    if ci.size == 0:
        return np.zeros_like(estimate)

    n_vars, n_outputs = estimate.shape
    if ci.ndim == 2:
        if ci.shape == (n_vars, 2):
            halfwidth = ((ci[:, 1] - ci[:, 0]) / 2.0).reshape(n_vars, 1)
            return np.repeat(halfwidth, n_outputs, axis=1)
        ci = ci.reshape(n_vars, n_outputs, 2)
    elif ci.ndim == 3:
        if ci.shape[0] == n_vars and ci.shape[1] == n_outputs and ci.shape[2] == 2:
            pass
        elif ci.shape[0] == n_vars and ci.shape[1] == 2 and ci.shape[2] == n_outputs:
            ci = np.moveaxis(ci, 1, 2)
        elif ci.shape[0] == 2 and ci.shape[1] == n_vars:
            ci = np.moveaxis(ci, 0, 2)
        else:
            raise RuntimeError(f"Formato inesperado do intervalo de confiança: {ci.shape}.")
    else:
        raise RuntimeError(f"Formato inesperado do intervalo de confiança: {ci.shape}.")
    return (ci[:, :, 1] - ci[:, :, 0]) / 2.0


def indices_from_analysis(analysis, n_outputs=None):
    results = analysis["Results"] if "Results" in analysis else analysis["results"]
    n_vars = problem["num_vars"]
    if n_outputs is None:
        n_outputs = len(time_points) - 1
    first_order = _as_var_by_output(results["FirstOrder"], n_vars, n_outputs)
    total = _as_var_by_output(results["Total"], n_vars, n_outputs)
    bootstrap = results.get("Bootstrap") or {}
    first_order_conf = _bootstrap_halfwidth(
        _first_order_bootstrap(bootstrap), first_order
    )
    total_conf = _bootstrap_halfwidth(bootstrap.get("Total"), total)

    # The initial state is deterministic, so its Sobol indices are undefined.
    def include_initial_timepoint(values):
        padded = np.full((n_vars, len(time_points)), np.nan)
        padded[:, 1:] = values
        return padded

    return {
        "S1": include_initial_timepoint(first_order),
        "ST": include_initial_timepoint(total),
        "S1_conf": include_initial_timepoint(first_order_conf),
        "ST_conf": include_initial_timepoint(total_conf),
    }


def save_convergence(output_dir, sample_sizes, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "convergencia.csv"
    plot_data = {}
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow((
            "N_anterior", "N_atual", "parametro",
            "RMSE_S1", "max_delta_S1", "S1_dentro_IC_combinado_pct",
            "RMSE_ST", "max_delta_ST", "ST_dentro_IC_combinado_pct",
        ))
        for previous_n, current_n in zip(sample_sizes, sample_sizes[1:]):
            previous = results[previous_n]
            current = results[current_n]
            plot_data[(previous_n, current_n)] = {"S1": [], "ST": []}
            for parameter_index, name in enumerate(problem['names']):
                row_values = []
                for index_name in ("S1", "ST"):
                    estimate_key = index_name
                    confidence_key = f"{index_name}_conf"
                    previous_estimate = previous[estimate_key][parameter_index]
                    current_estimate = current[estimate_key][parameter_index]
                    difference = np.abs(current_estimate - previous_estimate)
                    combined_confidence = np.hypot(
                        previous[confidence_key][parameter_index],
                        current[confidence_key][parameter_index],
                    )
                    valid = (
                        np.isfinite(difference)
                        & np.isfinite(combined_confidence)
                    )
                    if np.any(valid):
                        rmse = float(np.sqrt(np.mean(difference[valid] ** 2)))
                        max_difference = float(np.max(difference[valid]))
                        within_confidence = float(
                            100.0 * np.mean(
                                difference[valid] <= combined_confidence[valid]
                            )
                        )
                    else:
                        rmse = max_difference = within_confidence = float("nan")
                    row_values.extend((rmse, max_difference, within_confidence))
                    plot_data[(previous_n, current_n)][index_name].append(rmse)
                writer.writerow((
                    previous_n,
                    current_n,
                    name,
                    *row_values,
                ))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, index_name in zip(axes, ("S1", "ST")):
        for (previous_n, current_n), values in plot_data.items():
            ax.plot(
                problem['names'],
                values[index_name],
                marker="o",
                label=f"N={previous_n:,} → {current_n:,}",
            )
        ax.set_ylabel(f"RMSE temporal de {index_name}")
        ax.set_title(f"Convergência de {index_name} entre tamanhos de amostra")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Parâmetro")
    fig.tight_layout()
    fig.savefig(output_dir / "convergencia_indices.png", dpi=200)
    plt.close(fig)


DEFAULT_UQCLOUD_HOST = "https://uqcloud.ethz.ch"
DEFAULT_UQCLOUD_TOKEN = "7d56b9e352ade50a3b5a6426083315ea7f5c5e79"


def _open_uq_session(host, token):
    host = host or DEFAULT_UQCLOUD_HOST
    token = token or DEFAULT_UQCLOUD_TOKEN
    if not host or not token:
        raise RuntimeError(
            "Host/token do UQCloud ausentes. Passe --host e --token "
            "ou defina UQCLOUD_HOST e UQCLOUD_TOKEN."
        )
    session = sessions.cloud(host=host, token=token, log_level="WARNING")
    session.timeout = 86400
    session.reset()
    return session


def analyze_with_uqpylab(uq, N, seed):
    global _ACTIVE_OUTPUT_INDEX

    input_opts = {
        "Marginals": [
            {"Name": name, "Type": "Uniform", "Parameters": bounds}
            for name, bounds in zip(problem["names"], problem["bounds"])
        ]
    }
    uq.createInput(input_opts)
    indices = {
        key: np.full((problem["num_vars"], len(time_points)), np.nan)
        for key in ("S1", "ST", "S1_conf", "ST_conf")
    }
    try:
        # Scalar analyses avoid the UQCloud failure; cached trajectories prevent
        # rerunning the ODE model for every time point.
        for time_index in range(1, len(time_points)):
            _ACTIVE_OUTPUT_INDEX = time_index - 1
            uq.rng(seed, "twister")
            uq.createModel({
                "Type": "Model",
                "ModelFun": "sobol_uqpylab.evaluate_samples",
                "isVectorized": True,
                "CommFormat": "MAT",
            })
            analysis = uq.createAnalysis({
                "Type": "Sensitivity",
                "Method": "Sobol",
                "Sobol": {
                    "Order": 1,
                    "SampleSize": int(N),
                    "Sampling": "Sobol",
                },
                "Bootstrap": {
                    "Replications": 100,
                },
            })
            time_indices = indices_from_analysis(analysis, n_outputs=1)
            for key in indices:
                indices[key][:, time_index] = time_indices[key][:, 1]
        return indices
    finally:
        _ACTIVE_OUTPUT_INDEX = None
        _OUTPUT_CACHE_INDEX.clear()
        _OUTPUT_CACHE_VALUES.clear()


def main():
    parser = argparse.ArgumentParser(
        description="Análise MPI de sensibilidade de Sobol com UQPyLab."
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--N",
        type=int,
        help="Executa uma análise com N amostras base (útil para testes).",
    )
    sample_group.add_argument(
        "--Ns",
        type=int,
        nargs=3,
        metavar=("N1", "N2", "N3"),
        help="Executa três análises crescentes e compara sua convergência.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "resultados_sobol_uqpylab",
        help="Diretório para salvar o CSV e os gráficos.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("UQCLOUD_HOST", DEFAULT_UQCLOUD_HOST),
        help="URL do UQCloud.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("UQCLOUD_TOKEN", DEFAULT_UQCLOUD_TOKEN),
        help="Token do UQCloud.",
    )
    args = parser.parse_args()
    if args.N is not None and args.N < 2:
        parser.error("--N deve ser pelo menos 2.")
    if args.Ns is not None and (
        any(N < 2 for N in args.Ns)
        or args.Ns[0] >= args.Ns[1]
        or args.Ns[1] >= args.Ns[2]
    ):
        parser.error("--Ns deve conter três inteiros crescentes, todos >= 2.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    seed = 2026
    if args.N is not None:
        sample_sizes = (args.N,)
    elif args.Ns is not None:
        sample_sizes = tuple(args.Ns)
    else:
        sample_sizes = (32768, 65536, 131072)
    results = {}

    if rank == 0 and any(N & (N - 1) for N in sample_sizes):
        print(
            "Aviso: os N escolhidos não são potências de 2. "
            "A análise será executada com esses tamanhos, mas sequências Sobol "
            "têm melhor balanceamento com N potência de 2."
        )

    if rank != 0:
        _worker_loop()
        return

    session = None
    try:
        session = _open_uq_session(args.host, args.token)
        uq = session.cli
        for N in sample_sizes:
            total_simulations = (
                N
                * (problem["num_vars"] + 2)
                * (len(time_points) - 1)
            )
            _progress.update({
                "completed": 0,
                "total": total_simulations,
                "start": time.perf_counter(),
                "N": N,
            })
            print(
                f"Iniciando análise N={N:,}: {total_simulations:,} "
                f"simulações-modelo em {size} processo(s) MPI "
                f"({len(time_points) - 1} análises temporais no UQpyLab)..."
            )
            indices = analyze_with_uqpylab(uq, N, seed)
            if _progress["completed"]:
                print(flush=True)
            save_results(args.outdir / f"N_{N}", indices)
            results[N] = indices
            print(
                f"Análise N={N:,} concluída; resultados em "
                f"{(args.outdir / f'N_{N}').resolve()}"
            )
            session.reset()

        if len(sample_sizes) > 1:
            save_convergence(args.outdir, sample_sizes, results)
            print(f"Comparação de convergência salva em: {args.outdir.resolve()}")
    finally:
        _stop_workers()
        if session is not None:
            try:
                session.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
