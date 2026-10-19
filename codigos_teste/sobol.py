import matplotlib.pyplot as plt
import numpy as np

params = ['πv', 'kv1', 'kv2', 'αb', 'βps', 'βpl', 'βbm', 'δpl', 'B0', 'πb2']
n_labels = ['10k', '20k', '30k', '40k', '50k', '60k']

first_order = [
    [0.02983825, 0.03688777, 0.04681293, 0.03275573, 0.05566456, 0.04775115],
    [0.00937439, 0.05141187, 0.01733007, 0.03134425, 0.04410769, 0.03273295],
    [-0.00317952, 0.03973068, 0.0015981,  0.00491004, 0.0252548,  0.01151789],
    [-0.00696308, 0.01258588, 0.00392579, 0.00304938, 0.01571409, 0.01135364],
    [-0.00684602, 0.01021455,-0.00543614,-0.00456166, 0.01152101, 0.00000645],
    [-0.00710695, 0.00996731,-0.00658795,-0.00413705, 0.01241806,-0.00047208],
    [-0.00707944, 0.0099816, -0.00667331,-0.00416212, 0.01241737,-0.00057388],
    [-0.00708054, 0.00999486,-0.00667172,-0.00416136, 0.01243191,-0.00057052],
    [0.0014302,  0.01301401, 0.00403384, 0.01175794, 0.03653364, 0.00579829],
    [-0.00708165, 0.00999575,-0.00667132,-0.00416159, 0.01243169,-0.00057080]
]

total_order = [
    [0.67084143, 0.77880663, 0.76228694, 0.71065176, 0.75135353, 0.70719093],
    [0.59377351, 0.66789320, 0.59817849, 0.53072251, 0.60084520, 0.61378044],
    [0.52228666, 0.50445098, 0.47386647, 0.47633780, 0.52220048, 0.48790312],
    [0.16691246, 0.31438518, 0.32512269, 0.30693344, 0.33782803, 0.39301735],
    [-0.06186603, 0.01003147,-0.04101796,-0.09011128,-0.01188338,-0.01886649],
    [-0.06836069, 0.00052225,-0.04280787,-0.09521644,-0.02170194,-0.02464052],
    [-0.06620127,-0.00011219,-0.04398017,-0.09535316,-0.02256279,-0.02418024],
    [-0.06631069,-0.00023474,-0.04400336,-0.09545760,-0.02257030,-0.02424051],
    [0.48440186, 0.40795465, 0.42007354, 0.36800098, 0.41491211, 0.38442964],
    [-0.06631281,-0.00023638,-0.04400479,-0.09545606,-0.02256913,-0.02423898]
]

# --- Gráfico 1: Convergência 1ª ordem ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
for i, p in enumerate(params):
    ax1.plot(n_labels, first_order[i], marker='o', label=p, linewidth=1.5, markersize=4)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax1.set_title('Índices de 1ª ordem — convergência')
ax1.set_xlabel('n amostras')
ax1.set_ylabel('Índice de 1ª ordem')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# --- Gráfico 2: Convergência ordem total ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
for i, p in enumerate(params):
    ax2.plot(n_labels, total_order[i], marker='o', label=p, linewidth=1.5, markersize=4)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax2.set_title('Índices de ordem total — convergência')
ax2.set_xlabel('n amostras')
ax2.set_ylabel('Índice de ordem total')
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# --- Gráfico 3: Ranking em n=60k ---
fig3, ax3 = plt.subplots(figsize=(7, 5))
vals_60k = [max(0, total_order[i][-1]) for i in range(len(params))]
sorted_idx = np.argsort(vals_60k)
sorted_params = [params[i] for i in sorted_idx]
sorted_vals = [vals_60k[i] for i in sorted_idx]
ax3.barh(sorted_params, sorted_vals, color='steelblue', edgecolor='white')
ax3.set_title('Ranking — ordem total (n=60k)')
ax3.set_xlabel('Índice de ordem total')
ax3.grid(True, axis='x', alpha=0.3)
plt.tight_layout()

plt.show()