"""
Fig6: rMD17 prediction-error visualization (Baseline vs EDG++)
3 panels: Naphthalene, Uracil, Malonaldehyde
Style: matching reference image — box frame, no ticks, no shading, legend upper-right
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import uniform_filter1d
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw

matplotlib.rcParams.update({
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0,     # no tick marks
    'ytick.major.width':  0,
    'xtick.major.size':   0,     # no tick protrusions
    'ytick.major.size':   0,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

OUTPUT_DIR = '/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp/experiments'

C_BL = '#999999'    # gray for baseline
C_PP = '#E07B39'    # orange for EDG++
SMOOTH = 20

BASELINE_DIR = '/home/lzeng/workspace/GEOM3D/experiments/run_rMD17_baseline'
DISTILL_DIR  = '/home/lzeng/workspace/GEOM3D/experiments/run_rMD17_distillation/SphereNet/e1000_b128_lr5e-4_ed128_lsCosine/ED_E@mean_std'

MOLECULES = [
    dict(name='Naphthalene', title='SphereNet on Naphthalene',
         smiles='c1ccc2ccccc2c1',
         baseline=f'{BASELINE_DIR}/naphthalene/evaluation_best.pth.npz',
         edgpp=f'{DISTILL_DIR}/ED0.001_E@asb0_asa1.5_bb0/rs42/naphthalene/evaluation_best.pth.npz',
         mol_zoom=0.18),
    dict(name='Uracil', title='SphereNet on Uracil',
         smiles='O=c1cc[nH]c(=O)[nH]1',
         baseline=f'{BASELINE_DIR}/uracil/evaluation_best.pth.npz',
         edgpp=f'{DISTILL_DIR}/ED0.001_E@asb0_asa1.5_bb0/rs42/uracil/evaluation_best.pth.npz',
         mol_zoom=0.16),
    dict(name='Malonaldehyde', title='SphereNet on Malonaldehyde',
         smiles='O=CC=CO',
         baseline=f'{BASELINE_DIR}/malonaldehyde/evaluation_best.pth.npz',
         edgpp=f'{DISTILL_DIR}/ED0.001_E@asb0_asa1.0_bb0/rs42/malonaldehyde/evaluation_best.pth.npz',
         mol_zoom=0.18),
]


def smiles_to_image(smiles, size=(200, 200)):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=size)
    img = img.convert('RGBA')
    data = np.array(img)
    white = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
    data[white, 3] = 0
    return Image.fromarray(data)


def load_errors(path):
    import torch, io
    # Patch torch.load to force CPU for CUDA-saved tensors
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'map_location': 'cpu', 'weights_only': False})
    try:
        d = np.load(path, allow_pickle=True)['test_eval_dict'].item()
    finally:
        torch.load = _orig
    yt, yp = d['y_energy_true'], d['y_energy_pred']
    if hasattr(yt, 'cpu'): yt = yt.cpu().numpy()
    if hasattr(yp, 'cpu'): yp = yp.cpu().numpy()
    return np.abs(yt - yp)


def plot_panel(ax, mol):
    err_bl = load_errors(mol['baseline'])
    err_pp = load_errors(mol['edgpp'])

    bl_s = uniform_filter1d(err_bl, size=SMOOTH)
    pp_s = uniform_filter1d(err_pp, size=SMOOTH)
    x = np.arange(len(err_bl))

    # Lines only, no shading
    ax.plot(x, bl_s, color=C_BL, lw=0.7, alpha=0.9, zorder=3)
    ax.plot(x, pp_s, color=C_PP, lw=0.7, alpha=0.9, zorder=4)

    # Title above
    ax.set_title(mol['title'], fontsize=9, pad=5)

    # Axes labels
    ax.set_xlabel('Dynamic trajectories', fontsize=8)
    ax.set_ylabel(r'abs($y_{\mathrm{true}} - y_{\mathrm{pred}}$)', fontsize=8)

    # Range
    ax.set_xlim(0, len(x) - 1)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ymax = max(bl_s.max(), pp_s.max()) * 1.40
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # No per-panel legend (shared at bottom)

    # Molecule structure upper-left
    mol_img = smiles_to_image(mol['smiles'])
    imagebox = OffsetImage(mol_img, zoom=mol.get('mol_zoom', 0.18))
    ab = AnnotationBbox(imagebox, (0.15, 0.85),
                        xycoords='axes fraction', frameon=False, zorder=10)
    ax.add_artist(ab)

    mae_bl, mae_pp = err_bl.mean(), err_pp.mean()
    imp = (1 - mae_pp / mae_bl) * 100
    print(f'  {mol["name"]}: BL={mae_bl:.4f}, EDG++={mae_pp:.4f}, Δ={imp:+.1f}%')


def main():
    print('Generating Fig6 ...')
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75))

    for ax, mol in zip(axes, MOLECULES):
        plot_panel(ax, mol)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave room for legend

    # ── Shared legend at bottom (IJCAI style: colour patches) ──
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=C_BL, alpha=0.7, label='w/o EDG++'),
        Patch(facecolor=C_PP, alpha=0.7, label='w/ EDG++'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01),
               handlelength=1.8, handletextpad=0.5, columnspacing=2.5)

    for ext in ('pdf', 'png'):
        plt.savefig(f'{OUTPUT_DIR}/rMD17_baseline_vs_edgpp.{ext}',
                    bbox_inches='tight', dpi=600)
    plt.close()
    print('Saved rMD17_baseline_vs_edgpp.pdf / .png')


if __name__ == '__main__':
    main()
