import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer
import warnings

from modelo_acoplado_novo import coupled_model_wrapper

_S = 0.25    
_SF = 0.1     
_NOMINAL = {
    'πv': 6.80e-1, 'kv1': 4.82e-5, 'πps': 2e-3, 'kv2': 7.48e-7,
    'δps': 2.0e0, 'αb': 6.0e0, 'H1': 1e-4, 'f': 2.0, 'c': 0.2, 'r': 1.0
}
PARAM_NAMES = list(_NOMINAL.keys())

problem = {
    'num_vars': len(PARAM_NAMES),
    'names': PARAM_NAMES,
    'bounds': [[np.log(_SF * _NOMINAL[k]), _S] for k in PARAM_NAMES],
    'dists': ['lognorm'] * len(PARAM_NAMES),
}

N_TIME = 100 
PASTA_SAIDA = "./validacao_csv"
os.makedirs(PASTA_SAIDA, exist_ok=True)

def experimento_1(N_teste=64):
    print(f"\n--- INICIANDO EXPERIMENTO 1 (N={N_teste}) ---")

    print("1) Gerando amostras Sobol...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = sobol_sampler.sample(problem, N_teste, calc_second_order=False, seed=42)
    
    df_amostras = pd.DataFrame(X, columns=PARAM_NAMES)
    caminho_amostras = os.path.join(PASTA_SAIDA, "1_amostras_sobol.csv")
    df_amostras.to_csv(caminho_amostras, index=False)
    print(f"   -> Salvo em: {caminho_amostras} (Total: {len(X)} avaliações)")

    print("\n2) Lendo CSV e rodando o modelo (uma a uma)...")
    df_lido = pd.read_csv(caminho_amostras)
    
    saidas = np.zeros((len(df_lido), N_TIME))
    for i, row in df_lido.iterrows():
        saidas[i, :] = coupled_model_wrapper(row.values)
        if (i+1) % 200 == 0:
            print(f"   Progresso: {i+1}/{len(df_lido)}")

    colunas_tempo = [f"dia_{int(t)}" for t in np.linspace(0, 500, N_TIME)]
    df_saidas = pd.DataFrame(saidas, columns=colunas_tempo)
    caminho_saidas = os.path.join(PASTA_SAIDA, "2_saidas_modelo.csv")
    df_saidas.to_csv(caminho_saidas, index=False)
    print(f"   -> Salvo em: {caminho_saidas}")

    print("\n3) Lendo o CSV de saídas e calculando Sobol (S1)...")
    df_resultados = pd.read_csv(caminho_saidas)
    Y_matriz = df_resultados.values
    
    somas_S1 = []
    
    for t_idx in range(N_TIME):
        Si = sobol_analyzer.analyze(
            problem, Y_matriz[:, t_idx], 
            calc_second_order=False, 
            print_to_console=False
        )
        soma_atual = np.sum(Si["S1"])
        somas_S1.append(soma_atual)
        
    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 500, N_TIME), somas_S1, marker='.', color='purple')
    plt.axhline(1.0, color='red', linestyle='--', label='Referência (1.0)')
    plt.title("Soma dos Índices de Primeira Ordem (Σ S1) ao longo do tempo")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Σ S1")
    plt.ylim(0, 1.2)
    plt.grid(True)
    plt.legend()
    caminho_grafico_soma = os.path.join(PASTA_SAIDA, "3_soma_S1.png")
    plt.savefig(caminho_grafico_soma)
    plt.close()
    
    print(f"   -> Soma inicial de S1 (dia 0): {somas_S1[0]:.4f}")
    print(f"   -> Soma final de S1 (dia 500): {somas_S1[-1]:.4f}")
    print(f"   -> Gráfico da soma salvo em: {caminho_grafico_soma}")


def experimento_2():
    print("\n--- INICIANDO EXPERIMENTO 2 ---")
    print("Verificando se o que chega e o que sai faz sentido fisicamente...")
    
    caminho_amostras = os.path.join(PASTA_SAIDA, "1_amostras_sobol.csv")
    caminho_saidas = os.path.join(PASTA_SAIDA, "2_saidas_modelo.csv")
    
    if not os.path.exists(caminho_amostras) or not os.path.exists(caminho_saidas):
        print("Erro: Rode o Experimento 1 primeiro para gerar os CSVs.")
        return

    df_in = pd.read_csv(caminho_amostras)
    df_out = pd.read_csv(caminho_saidas)
    
    indices_teste = [0, len(df_in)//2, len(df_in)-1]
    
    t_vals = np.linspace(0, 500, N_TIME)
    plt.figure(figsize=(10, 6))
    
    for idx in indices_teste:
        parametros = df_in.iloc[idx].to_dict()
        curva_saida = df_out.iloc[idx].values
        
        label = f"Amostra {idx} (πv={parametros['πv']:.2e}, r={parametros['r']:.2f})"
        plt.plot(t_vals, curva_saida, label=label, linewidth=2)
        
        print(f"\n[Amostra {idx}]")
        print("  ENTRADA (Alguns Parâmetros):", {k: f"{parametros[k]:.2e}" for k in ['πv', 'r', 'αb']})
        print(f"  SAÍDA (Pico Máximo): {np.max(curva_saida):.2e} no dia {t_vals[np.argmax(curva_saida)]:.1f}")

    plt.title("Validação Visual das Saídas (Experimento 2)")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Variável de Saída (ex: Vírus ou Anticorpos)")
    plt.grid(True)
    plt.legend()
    
    caminho_grafico_exp2 = os.path.join(PASTA_SAIDA, "4_validacao_curvas.png")
    plt.savefig(caminho_grafico_exp2)
    plt.close()
    print(f"\n   -> Gráfico das curvas salvo em: {caminho_grafico_exp2}")


if __name__ == "__main__":
    experimento_1(N_teste=128)
    experimento_2()