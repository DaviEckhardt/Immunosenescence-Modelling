import csv
from pathlib import Path
from secrets import randbits
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

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
    'num_vars': 5,
    'names': np.array(['c', 'g', 'B1', 'B2', 'H1']),
    'bounds': [
        [0.19, 0.21],        # Limites para c
        [0.19, 0.21],        # Limites para g
        [0.95e-4, 1.05e-4],  # Limites para B1
        [0.95e-9, 1.05e-9],  # Limites para B2
        [0.95e-10, 1.05e-10] # Limites para H1
    ]
}

# 4. Geração das amostras para as três verificações
N_VALUES = (10000,)
MAX_N = max(N_VALUES)
SEED = randbits(32) if rank == 0 else None
SEED = comm.bcast(SEED, root=0)
NUM_RESAMPLES = 1000
NUM_RESAMPLES_TIME = 100
param_values = sobol_sample.sample(
    problem, MAX_N, calc_second_order=False, seed=SEED
) if rank == 0 else None

# 5. Avaliação do Modelo (Wrapper)
y0 = [1.0, 1e6, 0, 1e6, 0, 5e5, 0, 2.5e5, 0, 0, 0, 150, 0.1, 0.0, 0.0]
t_span = (0, 500)
time_points = np.linspace(t_span[0], t_span[1], 101)
state_names = (
    'V', 'Ap', 'Apm', 'Thn', 'The', 'Tkn', 'Tke', 'B', 'Ps', 'Pl',
    'Bm', 'A', 'w', 'x', 'y_inf',
)
output_dir = Path(__file__).with_name('graficos_sobol')
run_id = uuid4().hex if rank == 0 else None
run_id = comm.bcast(run_id, root=0)
if rank == 0:
    output_dir.mkdir(exist_ok=True)
comm.Barrier()
rank_csv_path = output_dir / f'.saidas_equacoes_{run_id}_rank_{rank:05d}.csv'
equation_csv_path = output_dir / 'saidas_equacoes.csv'
equation_csv_header = (
    'simulacao', 'status', 'mensagem', *problem['names'], 'tempo',
    *state_names, 'variaveis_nao_finitas',
)
sample_counts = [
    MAX_N * (problem['num_vars'] + 2) // world_size
    + (rank < MAX_N * (problem['num_vars'] + 2) % world_size)
    for rank in range(world_size)
]
sample_displacements = np.cumsum([0, *sample_counts[:-1]]).tolist()
local_sample_count = sample_counts[rank]
local_param_values = np.empty((local_sample_count, problem['num_vars']))

comm.Scatterv(
    [
        param_values,
        [count * problem['num_vars'] for count in sample_counts],
        [offset * problem['num_vars'] for offset in sample_displacements],
        MPI.DOUBLE,
    ] if rank == 0 else None,
    local_param_values,
    root=0,
)

local_Y_time = np.empty((local_sample_count, len(time_points)))
local_Y_peak = np.empty(local_sample_count)
local_errors = []

if rank == 0:
    print(
        f"Iniciando {MAX_N * (problem['num_vars'] + 2):,} simulações "
        f"em {world_size} processos para "
        f"N={', '.join(f'{n:,}' for n in N_VALUES)} "
        f"(semente={SEED})..."
    )

with rank_csv_path.open('w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(equation_csv_header)
    for local_index, X in enumerate(local_param_values):
        simulation_index = sample_displacements[rank] + local_index
        p = base_params.copy()
        p['c'], p['g'], p['B1'], p['B2'], p['H1'] = X

        sol = solve_ivp(
            coupled_model_ivp,
            t_span,
            y0,
            args=(p,),
            method='LSODA',
            max_step=1.0,
            t_eval=time_points,
        )

        finite_mask = np.isfinite(sol.y)
        has_non_finite = not finite_mask.all()
        if not sol.success or has_non_finite:
            if not sol.success and has_non_finite:
                status = 'falha_solver_e_valores_nao_finitos'
            elif not sol.success:
                status = 'falha_solver'
            else:
                status = 'valores_nao_finitos'
            details = []
            if not sol.success:
                details.append(sol.message)
            if has_non_finite:
                bad_state_indices = np.flatnonzero((~finite_mask).any(axis=1))
                bad_state_names = [state_names[index] for index in bad_state_indices]
                details.append(
                    'Valores não finitos em: ' + ', '.join(bad_state_names)
                )
            local_errors.append(
                f"Simulação {simulation_index + 1}: {'; '.join(details)}"
            )
        else:
            status = 'ok'

        if sol.t.size:
            for time_value, state_values in zip(sol.t, sol.y.T):
                bad_values = [
                    state_names[index]
                    for index, value in enumerate(state_values)
                    if not np.isfinite(value)
                ]
                writer.writerow((
                    simulation_index + 1,
                    status,
                    sol.message,
                    *X,
                    time_value,
                    *state_values,
                    '|'.join(bad_values),
                ))
        else:
            writer.writerow((
                simulation_index + 1,
                status,
                sol.message,
                *X,
                '',
                *([np.nan] * len(state_names)),
                '|'.join(state_names),
            ))

        if not sol.success or has_non_finite:
            local_Y_time[local_index, :] = np.nan
            local_Y_peak[local_index] = np.nan
            continue

        x_values = sol.y[13, :]
        local_Y_time[local_index, :] = x_values
        local_Y_peak[local_index] = np.max(x_values)

comm.Barrier()
csv_error_message = None
if rank == 0:
    try:
        with equation_csv_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(equation_csv_header)
            for process_rank in range(world_size):
                process_csv_path = (
                    output_dir
                    / f'.saidas_equacoes_{run_id}_rank_{process_rank:05d}.csv'
                )
                with process_csv_path.open(
                    'r', newline='', encoding='utf-8'
                ) as process_csv_file:
                    reader = csv.reader(process_csv_file)
                    next(reader, None)
                    writer.writerows(reader)
                process_csv_path.unlink()
        print(f'Saídas das equações salvas em: {equation_csv_path}')
    except (OSError, csv.Error) as error:
        csv_error_message = f'Não foi possível salvar as saídas das equações: {error}'
csv_error_message = comm.bcast(csv_error_message, root=0)
if csv_error_message:
    raise RuntimeError(csv_error_message)

time_counts = [count * len(time_points) for count in sample_counts]
time_displacements = [
    offset * len(time_points) for offset in sample_displacements
]
if rank == 0:
    Y_time = np.empty((sum(sample_counts), len(time_points)))
    Y_peak = np.empty(sum(sample_counts))
else:
    Y_time = None
    Y_peak = None

comm.Gatherv(
    local_Y_time.ravel(),
    [
        Y_time,
        time_counts,
        time_displacements,
        MPI.DOUBLE,
    ] if rank == 0 else None,
    root=0,
)
comm.Gatherv(
    local_Y_peak,
    [Y_peak, sample_counts, sample_displacements, MPI.DOUBLE]
    if rank == 0 else None,
    root=0,
)
all_errors = comm.gather(local_errors, root=0)
error_message = None
if rank == 0:
    errors = [error for process_errors in all_errors for error in process_errors]
    if errors:
        error_message = '\n'.join(errors)
error_message = comm.bcast(error_message, root=0)
if error_message:
    raise RuntimeError(error_message)

# 6. Cálculo e gráficos dos índices para cada tamanho amostral
if rank == 0:
    variable_name = 'x'
    for N in N_VALUES:
        sample_count = N * (problem['num_vars'] + 2)
        print(f"\nVerificação com N={N:,} (2^{N.bit_length() - 1}):")

        Si = sobol.analyze(
            problem,
            Y_peak[:sample_count],
            calc_second_order=False,
            num_resamples=NUM_RESAMPLES,
            seed=SEED,
        )

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

        Si_time = {
            index: np.full((len(time_points), problem['num_vars']), np.nan)
            for index in ('S1', 'S1_conf', 'ST', 'ST_conf')
        }
        for time_index in range(len(time_points)):
            output = Y_time[:sample_count, time_index]
            if np.std(output) == 0.0:
                continue

            time_result = sobol.analyze(
                problem,
                output,
                calc_second_order=False,
                num_resamples=NUM_RESAMPLES_TIME,
                seed=SEED,
            )
            for index in Si_time:
                Si_time[index][time_index, :] = time_result[index]

        plots = [('geral', range(problem['num_vars']))]
        plots.extend(
            (name, [parameter_index])
            for parameter_index, name in enumerate(problem['names'])
        )

        sample_output_dir = output_dir / f'N_{N}'
        sample_output_dir.mkdir(exist_ok=True)
        s1_sum_by_time = np.sum(Si_time['S1'], axis=1)
        csv_path = sample_output_dir / 'soma_S1_tempo.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(('tempo', 'soma_S1'))
            writer.writerows(zip(time_points, s1_sum_by_time))

        print("Tempo | Soma dos índices S1")
        for time_value, s1_sum in zip(time_points, s1_sum_by_time):
            print(f"{time_value:.2f} | {s1_sum:.6f}")
        print(f"Soma dos índices S1 salva em: {csv_path}")

        for plot_name, parameter_indices in plots:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            for parameter_index in parameter_indices:
                name = problem['names'][parameter_index]
                s1 = Si_time['S1'][:, parameter_index]
                s1_conf = Si_time['S1_conf'][:, parameter_index]
                st = Si_time['ST'][:, parameter_index]
                st_conf = Si_time['ST_conf'][:, parameter_index]

                axes[0].plot(time_points, s1, label=name)
                axes[0].fill_between(time_points, s1 - s1_conf, s1 + s1_conf, alpha=0.15)
                axes[1].plot(time_points, st, label=name)
                axes[1].fill_between(time_points, st - st_conf, st + st_conf, alpha=0.15)

            axes[0].set_ylabel('Índice de primeira ordem (S1)')
            axes[1].set_ylabel('Índice de ordem total (ST)')
            axes[1].set_xlabel('Tempo')
            for axis in axes:
                axis.grid(True, alpha=0.3)
                axis.legend(title='Parâmetro')

            fig.suptitle(
                f'Índices de Sobol ao longo do tempo para {variable_name} - '
                f'{plot_name} (N={N:,})'
            )
            fig.tight_layout()
            output_path = sample_output_dir / f'sobol_indices_{plot_name}.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Gráfico salvo em: {output_path}")