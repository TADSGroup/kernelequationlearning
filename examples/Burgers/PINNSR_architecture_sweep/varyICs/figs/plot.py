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
# One marker per curve so the series stay distinguishable in grayscale even where the
# lines overlap; the star reads much smaller than the others at equal markersize.
MARKERS = {'2x64': '^', '3x128': 'o', '4x256': 's'}
MARKERSIZE = 6
BENCHMARK_MARKER = '*'
BENCHMARK_MARKERSIZE = 10
# A trained model whose u_error exceeds this is treated as diverged, same bucket as a
# true NaN crash -- legitimate NRMSE values in this sweep never approach 1, let alone this.
DIVERGED_THRESHOLD = 1.0

errors = np.load('../errors_archsweep.npy', allow_pickle=True).item()

# Reference curve: mean PINN-SR error (over 10 runs) from the main benchmark run
# (../../benchmark_varyICs/i_smpl_errors/e_ismpl_PINNSR.npy), used there with an
# 8x20 architecture. Hardcoded here rather than recomputed -- these are just the
# already-plotted mean values from benchmark_varyICs/figs/u_errors_varyIC.pdf and
# P_errors_varyIC.pdf, painted on top for reference. Not subject to DIVERGED_THRESHOLD
# masking -- it's a fixed reference line, not one of the swept architectures.
BENCHMARK_LABEL = r'8$\times$20'
BENCHMARK = {
    'u': [2.857738684117794, 2.8204502917826177, 2.7959949024021626,
          2.8285130515694616, 2.8101908955723047, 2.7994784779846666,
          2.8143752928823234, 2.809718842431903, 2.7992257218807937],
    'P': [1.6085985839366912, 1.6116261035203934, 1.4447898387908935,
          1.3252126544713974, 1.4122688114643096, 1.1584774896502494,
          1.5895607948303223, 1.560141858458519, 1.2773570813238622],
}


def diverged_mask(arch):
    u = np.array(errors[arch]['u'], dtype=float)
    return ~np.isfinite(u) | (u > DIVERGED_THRESHOLD)


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


def plot_metric(key, out_path, clip_diverged):
    plt.figure(figsize=(6, 2.5))
    for arch in ARCHS:
        vals = np.array(errors[arch][key], dtype=float)
        mask = diverged_mask(arch)
        plot_vals = np.where(mask, np.nan, vals) if clip_diverged else vals
        plt.plot(N_OBS, plot_vals, color='black', label=LABELS[arch],
                  linestyle=LINESTYLES[arch], marker=MARKERS[arch], markersize=MARKERSIZE)

    plt.plot(N_OBS, BENCHMARK[key], color='black', label=BENCHMARK_LABEL,
              linestyle='dashdot', marker=BENCHMARK_MARKER, markersize=BENCHMARK_MARKERSIZE)

    plt.yscale('log')
    ax = plt.gca()

    # Diverged points stay excluded from the lines, but are no longer flagged with an x at
    # the top of the axes; pin the limits to the legitimate data range all the same.
    if clip_diverged:
        ax.set_ylim(*ax.get_ylim())

    # keep the tight autoscaled range, just show more y-axis tick labels for readability
    set_log_ticks(ax)

    # Legend moved above the axes: with 4 series (the flat 8x20 reference line sits
    # near the top of the clipped view) an in-plot "upper right" box would overlap data.
    plt.legend(fontsize=13, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()


# Main plots: diverged points excluded from the lines (see the printout below for which)
plot_metric('u', 'u_errors_varyIC_archsweep.pdf', clip_diverged=True)
plot_metric('P', 'P_errors_varyIC_archsweep.pdf', clip_diverged=True)

# Raw/unclipped versions for comparison -- shows exactly how bad the divergence is
plot_metric('u', 'u_errors_varyIC_archsweep_raw.pdf', clip_diverged=False)
plot_metric('P', 'P_errors_varyIC_archsweep_raw.pdf', clip_diverged=False)

for arch in ARCHS:
    mask = diverged_mask(arch)
    if mask.any():
        print(f'{arch}: diverged at n_obs={np.array(N_OBS)[mask].tolist()}')

print('wrote u_errors_varyIC_archsweep.pdf, P_errors_varyIC_archsweep.pdf (+ _raw variants)')
