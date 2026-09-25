"""
Darcy composite figure:
  left  : 2x3 grid, top row u_1,u_2,u_3 and bottom row f_1,f_2,f_3
  right : 2x1 block, top operator-learning error, bottom in-distribution error
          (shared x axis, no legends)

Only needs numpy, matplotlib (with LaTeX) and matplotlib-label-lines.
Data comes from data.npz, produced once by extract_data.py.

    python make_figure.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- settings
OBS_PTS = 2              # which observation-count dataset the fields come from (2, 4 or 8)
SAMPLES = [0, 1, 2]      # which of the m=5 sampled functions to show (0-based)
OBS_PTS_LIST = [2, 4, 8]

# Layout (inches). Every row of the 2x4 arrangement has height = FIELD.
FIELD = 1.05             # side of each square u/f panel
GAP_FIELD = 0.16         # horizontal gap between field panels
GAP_ROW = 0.30           # vertical gap between top and bottom rows (both blocks)
GAP_BLOCK = 0.52         # gap between field block and error block (room for y tick labels)
ERR_W = 2.40             # width of the error panels
MARGIN_L, MARGIN_R = 0.30, 0.06
MARGIN_B, MARGIN_T = 0.22, 0.08

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

d = np.load(os.path.join(HERE, 'data.npz'))
M = d['M']

# ---------------------------------------------------------------- figure + axes
fig_w = MARGIN_L + 3 * FIELD + 2 * GAP_FIELD + GAP_BLOCK + ERR_W + MARGIN_R
fig_h = MARGIN_B + 2 * FIELD + GAP_ROW + MARGIN_T
fig = plt.figure(figsize=(fig_w, fig_h))


def add_axes(x, y, w, h):
    return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])


y_top = MARGIN_B + FIELD + GAP_ROW
y_bot = MARGIN_B
field_axes = [[add_axes(MARGIN_L + j * (FIELD + GAP_FIELD), y, FIELD, FIELD) for j in range(3)]
              for y in (y_top, y_bot)]
x_err = MARGIN_L + 3 * FIELD + 2 * GAP_FIELD + GAP_BLOCK
ax_ol = add_axes(x_err, y_top, ERR_W, FIELD)
ax_id = add_axes(x_err, y_bot, ERR_W, FIELD)
ax_ol.sharex(ax_id)

# ---------------------------------------------------------------- fields
xy_fine = d['xy_fine']
xy_coll = d['xy_coll']
u = d[f'u_{OBS_PTS}obs']
f = d[f'f_{OBS_PTS}obs']
xy_obs = d[f'xyobs_{OBS_PTS}obs']

# colour scale shared across all m sampled functions, as in the original figures
for row, vals in enumerate([u, f]):
    vmin, vmax = vals.min(), vals.max()
    for col, s in enumerate(SAMPLES):
        ax = field_axes[row][col]
        ax.tricontourf(*xy_fine.T, vals[s], levels=15, vmin=vmin, vmax=vmax)
        ax.scatter(*xy_coll.T, c='white', s=1.5, clip_on=False, edgecolors='none')
        if row == 0:
            ax.scatter(*xy_obs[s].T, c='red', s=9, alpha=0.8, clip_on=False, edgecolors='none')
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ticks = [0, 0.5, 1]
        labels = [r'$0$', r'$0.5$', r'$1$']
        # x tick labels only on the bottom row
        if row == 1:
            ax.set_xticks(ticks, labels)
        else:
            ax.set_xticks([])
        # y tick labels only on the left-most column
        if col == 0:
            ax.set_yticks(ticks, labels)
        else:
            ax.set_yticks([])

# ---------------------------------------------------------------- errors
def plot_errors(ax, tag, xvals):
    for short, ls, band in [('2step', 'dashed', 'red'), ('1step', 'solid', 'green')]:
        for n in OBS_PTS_LIST:
            k = f'{tag}_{short}_{n}'
            ax.plot(M, d[k + '_mean'], label=f'{n} pts', marker='o', markersize=2.5,
                    linestyle=ls, linewidth=0.9, color='black')
            ax.fill_between(M, d[k + '_lower'], d[k + '_upper'], alpha=.2, color=band,
                            linewidth=0)
    ax.set_yscale('log')
    ax.minorticks_off()
    labelLines(ax.get_lines(), align=True, xvals=xvals, fontsize=6.5)


plot_errors(ax_ol, 'ol', xvals=[80, 45, 100] + [80, 45, 100])
plot_errors(ax_id, 'id', xvals=[80, 45, 100] + [80, 45, 120])
ax_ol.set_ylim(top=1e0)

ax_id.set_xticks(M)
ax_id.set_xticklabels([r'$2$', '', '', r'$16$', r'$32$', r'$64$', r'$128$'])
ax_ol.tick_params(axis='x', which='both', labelbottom=False)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(HERE, f'darcy_composite.{ext}'), dpi=300)
print(f'figure size: {fig_w:.2f} x {fig_h:.2f} in')
