"""Plot cavity area evolution and final area versus subglacial water pressure.

The input files are selected explicitly to reproduce the figure from the study.
Each NPZ file must contain arrays named ``time`` (hours) and ``area`` (m^2).

Example (repository layout):
    python src/plot_AvsP.py

Example (cluster or external storage):
    python src/plot_AvsP.py --data-dir /path/to/simulation-data \
        --output-dir /path/to/figures

Expected subdirectories of --data-dir: ``luisa/`` and ``scratch/``.
These names identify the original data collections, not required cluster paths.
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
data_dir = args.data_dir.expanduser().resolve()
output_dir = args.output_dir.expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
luisa = data_dir / "luisa"
scratch = data_dir / "scratch"

# Consistent typography for the publication figures.
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

colors = [
    "darkslategrey", "teal", "lightseagreen", "mediumturquoise",
    "paleturquoise", "aquamarine",
]
velocity_label = [0.0, 0.2, 0.5, 0.7, 1.0]  # Basal velocities [m/day].

# Explicit selection of simulation runs (rows: velocities; columns: pressures).
# Pressures per row are 1, 3, 5, 10, and 20 MPa, respectively.

#TODO: Add .npz-files on github in one data folder; then replace luisa/ and scratch/ with that folder. 
#TODO: Did we use T=25d for some and T=30d for others? If so, we should make that consistent.

filenames = [
    [luisa / "production_49370437Run0_v0.0_p1.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370497Run0_v0.0_p3.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370501Run0_v0.0_p5.0e+06_T25_eta1.00e+13.npz",
    scratch / "production_49270157Run0_v0.0_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270152Run0_v0.0_p2.0e+07_T30_eta1.00e+13.npz"],

    [luisa / "production_49370436Run0_v0.2_p1.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370496Run0_v0.2_p3.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370502Run0_v0.2_p5.0e+06_T25_eta1.00e+13.npz",
    scratch / "production_49270156Run0_v0.2_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270143Run0_v0.2_p2.0e+07_T30_eta1.00e+13.npz"],

    [luisa / "production_49370435Run0_v0.5_p1.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370495Run0_v0.5_p3.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370500Run0_v0.5_p5.0e+06_T25_eta1.00e+13.npz",
    scratch / "production_49270155Run0_v0.5_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270142Run0_v0.5_p2.0e+07_T30_eta1.00e+13.npz"],

    [luisa / "production_49370434Run0_v0.7_p1.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370494Run0_v0.7_p3.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370498Run0_v0.7_p5.0e+06_T25_eta1.00e+13.npz",
    scratch / "production_49270154Run0_v0.7_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270140Run0_v0.7_p2.0e+07_T30_eta1.00e+13.npz"],

    [luisa / "production_49370433Run0_v1.0_p1.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370438Run0_v1.0_p3.0e+06_T25_eta1.00e+13.npz",
    luisa / "production_49370439Run0_v1.0_p5.0e+06_T25_eta1.00e+13.npz",
    scratch / "production_49270153Run0_v1.0_p1.0e+07_T30_eta1.00e+13.npz",
    scratch / "production_49270139Run0_v1.0_p2.0e+07_T30_eta1.00e+13.npz"]
]


# Verify that the explicitly selected data files are available before plotting.
missing = [path for group in filenames for path in group if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "Missing simulation files (configure --data-dir or provide the NPZ data):\n"
        + "\n".join(str(path) for path in missing)
    )

# Arrays used to compile the final-area plot across all velocities.
A_final = []
pressures = []

for k in range(5):
    # Store the last area measurement and its error for the summary plot.
    A_final_sub, pressures_sub = [], []
    plt.figure(figsize=(5, 3.75))

    # Standard plotting routine: one curve per water pressure.
    for i,filename in enumerate(filenames[k]):
        data = np.load(filename)

        area = data["area"]
        time = data["time"]/24

        # Read simulation parameters from the filename, not the directory path.
        params = filename.name.split('_')

        p = float(params[3][1:])*1e-6

        # Include pressure and velocity in the legend label.
        label = f"{params[3]}, {params[2]}"

        A_final_sub.append(area[-1])
        pressures_sub.append(p)

        plt.plot(time[:len(area)], area, ".", markersize=1, color=colors[i], label=label) 

    A_final.append(A_final_sub)
    pressures.append(pressures_sub)

    plt.xlabel("Time [d]")
    plt.ylabel(r"Area [m$^2$]")
    plt.yscale("log")
    plt.xscale("linear")
    plt.xlim(0,25)
    plt.legend(loc="upper left", ncols=2)
    plt.grid()
    plt.tight_layout()

    plt.savefig(output_dir / f"vary_pressure_v{velocity_label[k]:.1f}_final.pdf", dpi=300)
    plt.close()

# Summarize the final cavity area as a function of water pressure for each velocity.
fig, ax = plt.subplots(figsize=(5, 3.75))
markers = ["o", "v", "s", "D", "^"]
for k, velocity in enumerate(velocity_label):
    ax.plot(
        pressures[k], A_final[k],
        marker=markers[k], markersize=5, linestyle="-",
        color=colors[k], label=f"v = {velocity:.1f} m/d",
    )

ax.set_ylabel(r"Final area [m$^2$]")
ax.set_xlabel("Water pressure [MPa]")
ax.set_yscale("log")
ax.legend(loc="lower right", reverse=True)
plt.tight_layout()
plt.savefig(output_dir / "pressure_vs_area_final.pdf", dpi=300)
plt.close()