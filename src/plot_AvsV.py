"""Plot cavity area evolution and final area versus horizontal ice velocity.

The input files are selected explicitly to reproduce the figure from the study.
Each NPZ file must contain arrays named ``time`` (hours) and ``area`` (m^2).

Example (repository layout):
    python src/plot_AvsV.py

Example (cluster or external storage):
    python src/plot_AvsV.py --data-dir /path/to/simulation-data \
        --output-dir /path/to/figures
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse portable input/output paths without depending on the working directory."""
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=repository_root / "data",
        help="Directory containing the luisa/ and scratch/ NPZ subdirectories.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repository_root / "output" / "plots",
        help="Directory for generated PDF figures.",
    )
    return parser.parse_args()


args = parse_args()
data_dir = "../data/cluster"
luisa = data_dir / "Luisa"
jonas = data_dir / "jonas"

# Consistent typography for the publication figures.
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

colors = ['darkslategrey', 'teal', 'lightseagreen', 'mediumturquoise', 'paleturquoise', 'aquamarine']
pressure_label = [1e6, 3e6, 5e6, 1e7, 2e7]  # Water pressures [Pa].
linestyles = ["-", "--", ":", "-", "--", ":"]

# Explicit selection of simulation runs (rows: pressures; columns: velocities).
filenames = [
    [
    luisa /  "production_49370433Run0_v1.0_p1.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370434Run0_v0.7_p1.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370435Run0_v0.5_p1.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370436Run0_v0.2_p1.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370437Run0_v0.0_p1.0e+06_T25_eta1.00e+13.npz"
    ],

    [
    luisa /  "production_49370438Run0_v1.0_p3.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370494Run0_v0.7_p3.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370495Run0_v0.5_p3.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370496Run0_v0.2_p3.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370497Run0_v0.0_p3.0e+06_T25_eta1.00e+13.npz"
    ],

    [
    luisa /  "production_49370439Run0_v1.0_p5.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370498Run0_v0.7_p5.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370500Run0_v0.5_p5.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370502Run0_v0.2_p5.0e+06_T25_eta1.00e+13.npz", 
    luisa /  "production_49370501Run0_v0.0_p5.0e+06_T25_eta1.00e+13.npz"
    ],

    [scratch / "production_49270153Run0_v1.0_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270154Run0_v0.7_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270155Run0_v0.5_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270156Run0_v0.2_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270157Run0_v0.0_p1.0e+07_T30_eta1.00e+13.npz"],

    [scratch / "production_49270139Run0_v1.0_p2.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270140Run0_v0.7_p2.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270142Run0_v0.5_p2.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270143Run0_v0.2_p2.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270152Run0_v0.0_p2.0e+07_T30_eta1.00e+13.npz"],
]

# Store the last simulated area and the velocity for each run.
# These are end-of-run values.
A_final = []
velocities = []

for k, pressure_files in enumerate(filenames):
    A_final_sub, velocities_sub = [], []
    plt.figure(figsize=(5, 3.75))
    for i, filename in enumerate(pressure_files):
        with np.load(filename) as data:
            area = data["area"]
            time = data["time"] / 24  # Convert hours to days.

        # Extract the horizontal velocity from the filename, not the directory.
        params = filename.name.split('_')
        v = float(params[2][1:])
        label = f"{v:.1f} m/d"
        A_final_sub.append(area[-1])
        velocities_sub.append(v)
        plt.plot(time[:len(area)], area, markersize=1, color=colors[i],
                 linestyle=linestyles[i], label=label)

    A_final.append(A_final_sub)
    velocities.append(velocities_sub)
    plt.xlabel("Time [d]")
    plt.ylabel(r"Area [m$^2$]")
    plt.yscale("linear")
    plt.xscale("linear")
    plt.text(0.99, 0.97, f"p = {pressure_label[k]/1e6:.0f} MPa",
             ha="right", va="top", transform=plt.gca().transAxes)
    plt.xlim(0, 25)
    plt.legend(loc="lower left" if k < 3 else "upper left", ncols=1)
    plt.tight_layout()
    plt.savefig(output_dir / f"vary_velocity_p{pressure_label[k]:.0e}_final.pdf", dpi=300)
    plt.close()

fig, axes = plt.subplots(2,1, figsize=(5,3.75), sharex=True)
markers = ["o", "v", "s", "D", "^"]

# Upper panel: high water pressures (10 and 20 MPa).
for l in [3,4]:
    axes[0].plot(
        velocities[l], A_final[l],
        marker=markers[l], markersize=5, linestyle="-",
        color=colors[l], label=f"p = {pressure_label[l]/1e6:.0f} MPa"
    )
axes[0].set_ylabel(r"Final area [m$^2$]")
axes[0].set_yscale("log")
axes[0].legend(loc="lower right")
axes[0].text(0.01, 0.97, "a", size="large", weight="bold", ha="left", va="top", transform=axes[0].transAxes)

# Lower panel: low water pressures (1, 3 and 5 MPa).
for l in range(3):
    axes[1].plot(
        velocities[l], A_final[l],
        marker=markers[l], markersize=5, linestyle="-",
        color=colors[l], label=f"p = {pressure_label[l]/1e6:.0f} MPa"
    )
axes[1].set_xlabel("Horizontal velocity [m/d]")
axes[1].set_ylabel(r"Final area [m$^2$]")
axes[1].legend(loc="lower right")
axes[1].text(0.01,0.97, "b", weight="bold", size="large", ha="left", va="top", transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(output_dir / "velocity_vs_area_final.pdf", dpi=300)
plt.close(fig)
