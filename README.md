# Transient Cavity Evolution in a Viscoplastic Glacier Model

This repository contains the research code for simulating the transient
evolution of a two-dimensional cavity beneath a glacier using the
finite-element method.

It is associated with the paper: "Investigating Subglacial Cavity Evolution Using
Two-Dimensional Viscoelastic Ice Sheet Modeling"

## Requirements

The model is implemented using FEniCSx/DOLFINx and relies on the
following scientific Python libraries:

- FEniCSx/DOLFINx
- Basix
- UFL
- PETSc and petsc4py
- MPI and mpi4py
- NumPy
- SciPy
- h5py



## Quickstart

### 1. Prepare the computational environment

Install or activate a compatible FEniCSx/DOLFINx environment with the required scientific Python libraries.

### 2. Prepare the mesh

The simulation requires a two-dimensional mesh in XDMF format, including facet tags for the physical boundaries.
The mesh geometry and refinement can be adjusted.

These parameters define the computational domain and must match the
corresponding parameters in `src/2D_cavity.py`.

| Parameter | Description |
|---|---|
| `step_height` | Basal step height [m] |
| `step_length` | Basal step length [m] |
| `ice_height` | Total height of the ice domain [m] |
| `ice_length` | Total horizontal length of the domain [m] |
| `h_size_around_cavity` | Target element size near the cavity [m] |
| `h_size_around_cavity_fine` | Target element size in the ultra-fine region near the cavity [m] |
| `h_size_around_top` | Target element size near the upper and lower boundaries [m] |
| `h_size_bulk` | Target element size in the bulk domain [m] |

### 3. Configure the simulation

Check the parameters and file paths in `src/2D_cavity.py` before running the model.


| Parameter | Description |
|---|---|
| `step_height` | Basal step height [m] — must match the mesh geometry |
| `step_length` | Basal step length [m] — must match the mesh geometry |
| `ice_height` | Ice thickness [m] — must match the mesh geometry |
| `ice_length` | Total ice length [m] — must match the mesh geometry |
| `p_water` | Water pressure [Pa] |
| `v_constant_in` | Prescribed inflow velocity [m/day] |
| `v_constant_out` | Prescribed outflow velocity [m/day] |
| `T_end` | Simulation duration [days] |
| `dt_SI` | Time-step size [s] |

Adjust the mesh and output paths to match your execution environment.

### 4. Run the simulation

For local execution, the command-line interface accepts:

```bash
python3 src/2D_cavity.py --is_local true
```

For cluster execution, a job ID and a user ID must be supplied:

```bash
python3 src/2D_cavity.py --jobid JOB_ID --username USER_ID
```

### 5. Visualize the result e.g. using paraview