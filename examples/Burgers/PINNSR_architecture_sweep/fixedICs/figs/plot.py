import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 20
})

N_OBS = [10, 30, 50, 100, 200, 300, 400, 500, 600]
ARCHS = ['2x64', '3x128', '4x256']
LABELS = {'2x64': r'2$\times$64', '3x128': r'3$\times$128', '4x256': r'4$\times$256'}
LINESTYLES = {'2x64': 'solid', '3x128': 'dashed', '4x256': 'dotted'}

errors = np.load('../errors_archsweep.npy', allow_pickle=True).item()


def set_log_ticks(ax, n_labels=4):
    '''Explicit "nice" log-scale tick labels within the current (tight) ylim --
    matplotlib's LogFormatter silently drops all labels when the view doesn't
    happen to contain an exact power of ten, so we place them by hand instead.'''
    bottom, top = ax.get_ylim()
    candidates = []
    decade = int(np.floor(np.log10(bottom)))
    while 10**decade <= top:
        for m in [1, 2, 3, 5]:
            val = m * 10**decade
            if bottom <= val <= top:
                candidates.append(val)
        decade += 1
    candidates = sorted(set(candidates))
    if len(candidates) > n_labels:
        idx = sorted(set(np.linspace(0, len(candidates) - 1, n_labels).round().astype(int)))
        candidates = [candidates[i] for i in idx]
    labels = []
    for v in candidates:
        exp = int(np.floor(np.log10(v) + 1e-9))
        mant = v / 10**exp
        if abs(mant - 1) < 1e-6:
            labels.append(rf'$10^{{{exp}}}$')
        else:
            mant_str = f'{mant:.0f}' if abs(mant - round(mant)) < 1e-6 else f'{mant:.1f}'
            labels.append(rf'${mant_str}\times10^{{{exp}}}$')
    ax.set_yticks(candidates)
    ax.set_yticklabels(labels)


def plot_metric(key, out_path):
    plt.figure(figsize=(6, 2.5))
    for arch in ARCHS:
        plt.plot(N_OBS, errors[arch][key], color='black', label=LABELS[arch],
                  linestyle=LINESTYLES[arch], marker='o', markersize=4)
    plt.yscale('log')
    set_log_ticks(plt.gca())
    plt.legend(fontsize=13, loc='upper right', ncol=3)
    plt.xticks([])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()


plot_metric('u', 'u_errors_fixedIC_archsweep.pdf')
plot_metric('P', 'P_errors_fixedIC_archsweep.pdf')

print('wrote u_errors_fixedIC_archsweep.pdf and P_errors_fixedIC_archsweep.pdf')
