import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

outdir = os.path.join(os.path.dirname(__file__), "inflammaging_figures")
os.makedirs(outdir, exist_ok=True)

# --- Equações do modelo ---
def matsuo_ode(y, t, params):
    
    w, x, y_inf, z = y
    # parametros
    a = params['a']   # produção de inflamação por dano
    b = params['b']   # dano por atividade imune
    c = params['c']   # dano por patógeno
    d = params['d']   # decaimento de inflamação (adição + auto-inibição e)
    e = params['e']   # auto-inibição da inflamação
    f = params['f']   # efeito de y sobre w (aumenta imune)
    g = params['g']   # recuperação de dano
    h = params['h']   # eficácia imune em atacar patógeno
    k = params['k']   # decaimento de w
    r = params['r']   # taxa de crescimento do patógeno
    s = params.get('s', 0.0)  # ativação imunológica não-inflamatória (pode ser 0)

    # Eqs:
    dw_dt = s + f * y_inf - k * w                 
    dx_dt = b * w + c * z - g * x                 
    dy_dt = a * x - (d + e * y_inf) * y_inf       
    dz_dt = (r - h * w) * z                       

    return [dw_dt, dx_dt, dy_dt, dz_dt]

# --- Parâmetros de referência usados nas figuras do artigo ---
default_params = {
    'a': 0.2,
    'b': 0.2,
    'c': 0.2,
    'd': 0.08,   
    'e': 0.2,
    'f': 2.0,
    'g': 0.2,
    'h': 2.0,
    'k': 2.0,
    'r': 1.0,
    's': 0.0     
}

def integrate(params, y0, t):
    sol = odeint(matsuo_ode, y0, t, args=(params,))
    return sol

def plot_examples():
    ts = np.linspace(0, 400, 4001)   
    # condições iniciais: pequeno foco de patógeno e zero em w,x,y
    y0 = [0.1, 0.0, 0.0, 0.01]   # w,x,y,z 
    ds = [0.08, 0.16, 0.205]     # valores usados no artigo 
    fig, axes = plt.subplots(len(ds), 3, figsize=(12, 3.5*len(ds)))
    for i, dval in enumerate(ds):
        p = default_params.copy()
        p['d'] = dval
        sol = integrate(p, y0, ts)
        w = sol[:,0]; x = sol[:,1]; y_inf = sol[:,2]; z = sol[:,3]

        # time series plot (pathogen)
        ax1 = axes[i,0]
        ax1.plot(ts, z, label=f'z (patógeno), d={dval}')
        ax1.set_xlabel("tempo")
        ax1.set_ylabel("z (patógeno)")
        ax1.grid(True)
        ax1.legend()

        # time series plot (inflammation)
        ax2 = axes[i,1]
        ax2.plot(ts, y_inf, label=f'y (inflamação), d={dval}')
        ax2.set_xlabel("tempo")
        ax2.set_ylabel("y (inflamação)")
        ax2.grid(True)
        ax2.legend()
        
        ax2 = axes[i,2]
        ax2.plot(ts, x, label=f'x (dano tecidual), d={dval}')
        ax2.set_xlabel("tempo")
        ax2.set_ylabel("x (dano tecidual)")
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_examples.png"), dpi=200)
    plt.show()
    print("Figuras de exemplo (Fig.2-style) salvas em:", outdir)

def main():
    print("Rodando exemplos (Fig.2)...")
    plot_examples()

if __name__ == "__main__":
    main()
