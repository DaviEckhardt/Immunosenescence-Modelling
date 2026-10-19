import argparse
import csv
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
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


def analyze_outputs(Y, seed):
    indices = {
        key: np.full(
            (problem['num_vars'], len(time_points)),
            np.nan,
        )
        for key in ("S1", "ST", "S1_conf", "ST_conf")
    }
    for time_index in range(len(time_points)):
        output_at_time = Y[:, time_index]
        if np.var(output_at_time) == 0.0:
            continue
        time_indices = sobol.analyze(
            problem,
            output_at_time,
            calc_second_order=False,
            num_resamples=100,
            seed=seed + time_index,
        )
        for key in indices:
            indices[key][:, time_index] = time_indices[key]
    return indices


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


def main():
    parser = argparse.ArgumentParser(description="Análise MPI de sensibilidade de Sobol.")
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
        default=Path(__file__).resolve().parent / "resultados_sobol",
        help="Diretório para salvar o CSV e os gráficos.",
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

    for N in sample_sizes:
        error_message = None
        if rank == 0:
            try:
                param_values = sobol_sample.sample(
                    problem,
                    N,
                    calc_second_order=False,
                    seed=seed,
                )
                chunks = np.array_split(param_values, size)
                print(
                    f"Iniciando análise N={N:,}: {len(param_values):,} "
                    f"simulações em {size} processo(s) MPI..."
                )
            except Exception as exc:
                chunks = None
                error_message = f"Falha ao gerar amostras Sobol para N={N}: {exc}"
        else:
            chunks = None

        error_message = comm.bcast(error_message, root=0)
        if error_message:
            raise RuntimeError(error_message)
        local_samples = comm.scatter(chunks, root=0)

        local_outputs = np.empty(
            (len(local_samples), len(time_points)),
            dtype=np.float64,
        )
        local_error = None
        total_simulations = N * (problem['num_vars'] + 2)
        progress_steps = min(100, max(1, total_simulations // size))
        local_batches = np.array_split(local_samples, progress_steps)
        completed_local = 0
        start_time = time.perf_counter()

        for batch in local_batches:
            if local_error is None:
                for X in batch:
                    try:
                        local_outputs[completed_local, :] = evaluate_sample(X)
                        completed_local += 1
                    except Exception as exc:
                        local_error = (
                            f"Rank {rank}, simulação local "
                            f"{completed_local + 1}, N={N}: {exc}"
                        )
                        break

            completed_global = comm.allreduce(completed_local, op=MPI.SUM)
            if rank == 0:
                progress = 100.0 * completed_global / total_simulations
                elapsed = time.perf_counter() - start_time
                print(
                    f"\rN={N:,} | progresso: {progress:6.2f}% "
                    f"({completed_global:,}/{total_simulations:,}) "
                    f"| decorrido: {elapsed / 60:.1f} min",
                    end="",
                    flush=True,
                )
        if rank == 0:
            print(flush=True)

        errors = comm.allgather(local_error)
        error_message = next((message for message in errors if message), None)
        if error_message:
            raise RuntimeError(error_message)

        if rank == 0:
            Y = np.empty(
                (N * (problem['num_vars'] + 2), len(time_points)),
                dtype=np.float64,
            )
            row_counts = [len(chunk) for chunk in chunks]
            counts = np.asarray(
                [count * len(time_points) for count in row_counts],
                dtype=np.int32,
            )
            displacements = np.zeros(size, dtype=np.int32)
            if size > 1:
                displacements[1:] = np.cumsum(counts[:-1])
            receive_spec = [Y, counts, displacements, MPI.DOUBLE]
        else:
            Y = None
            receive_spec = None

        comm.Gatherv(local_outputs, receive_spec, root=0)

        error_message = None
        if rank == 0:
            try:
                if not np.isfinite(Y).all():
                    raise RuntimeError(
                        f"A saída do modelo contém valores não finitos para N={N}."
                    )
                indices = analyze_outputs(Y, seed)
                save_results(args.outdir / f"N_{N}", indices)
                results[N] = indices
                print(
                    f"Análise N={N:,} concluída; resultados em "
                    f"{(args.outdir / f'N_{N}').resolve()}"
                )
            except Exception as exc:
                error_message = f"Falha na análise N={N}: {exc}"

        error_message = comm.bcast(error_message, root=0)
        if error_message:
            raise RuntimeError(error_message)

    if rank == 0 and len(sample_sizes) > 1:
        save_convergence(args.outdir, sample_sizes, results)
        print(f"Comparação de convergência salva em: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()