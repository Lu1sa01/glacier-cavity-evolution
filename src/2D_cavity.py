"""Transient cavity-evolution simulation for a viscoplastic glacier model.

This script solves a quasi-static, two-dimensional ice-flow problem with a
moving cavity boundary and a rate-dependent viscous response. The model is
implemented in FEniCS/Dolfinx and includes:

- a temperature-dependent Arrhenius-type viscosity law,
- an internal viscous strain variable that evolves in time,
- hydrostatic water-pressure loading on the cavity boundary,
- a penalty/Lagrange-type contact treatment near the cavity and bedrock,
- export of geometry and diagnostic fields to XDMF and NPZ outputs.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np 
import dolfinx as dlfx
from mpi4py import MPI
from basix.ufl import  element, quadrature_element
import ufl 
from scipy.interpolate import interpn 
import sys
from dolfinx.fem.petsc import LinearProblem 
import h5py
import os
import time
import argparse
from dolfinx.fem import locate_dofs_geometrical
from petsc4py import PETSc

# =============================================================================
# SIMULATION SETUP
# =============================================================================

runtime_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--is_local",
    type=lambda x: str(x).lower() == "true",
    default=False,
    help="Run the simulation locally instead of on the Lichtenberg cluster.",
)
parser.add_argument(
    "--jobid",
    type=str,
    default=None,
    help="Job ID used to distinguish output files on the cluster.",
)
parser.add_argument(
    "--username",
    type=str,
    default=None,
    help="TU Darmstadt user ID used for cluster paths.",
)

args = parser.parse_args()

if not args.is_local and args.jobid is None:
    parser.error("A job ID is required when running on the cluster.")



# =============================================================================
# FILE PATHS
# =============================================================================

# Name of the input mesh.
input_mesh = (
    ""
) #add input mesh

# Base name of the simulation output.
output_simulation = "production"



# Paths for local execution.
output_prefix = "/root/share/"
mesh_prefix = (
    "/root/share/"
    "Hydro/output/Mesh/testing/"
)
hdf5_filename = (
    "/root/share/"
    "Hydro/output/Simulation/production.h5"
)


# Paths for execution on the Lichtenberg cluster.
if not args.is_local:
    TU_ID = args.username

    output_prefix = f"/work/scratch/{TU_ID}/"
    mesh_prefix = (
        f"/home/{TU_ID}/"
        "NUIW_Simulations/Hydro/output/Mesh/readyforsims/"
    )
    hdf5_filename = (
        f"/home/{TU_ID}/"
        "NUIW_Simulations/Hydro/output/Simulation/production.h5"
    )

    output_simulation = f"{output_simulation}_{args.jobid}"


# Determine a unique run number for output to avoid overwriting previous simulations.
run = 0
while os.path.exists(
    output_prefix + output_simulation + str(run) + ".xdmf"
):
    run += 1

filename_mesh = mesh_prefix + input_mesh
output_filename = (
    output_prefix + output_simulation + "Run" + str(run)
)


# =============================================================================
# GEOMETRY
# =============================================================================

step_height = 1.0       # Basal step height [m]
step_length = 5.0       # Basal step length [m]
ice_height = 1000.0     # Ice thickness [m]
ice_length = 2000.0     # Total ice length [m]

ice_L = (ice_length-step_length)/2     



# =============================================================================
# PHYSICAL SIMULATION PARAMETERS
# =============================================================================


u_total_in = dlfx.fem.Constant(region, dlfx.default_scalar_type(0.0)) 
u_total_out = dlfx.fem.Constant(region, dlfx.default_scalar_type(0.0))


# Ice density and gravitational acceleration.
rho_ice = 918.0       # Ice density [kg/m^3]
g = 9.81               # Gravitational acceleration [m/s^2]

# Body force due to gravity.
f = dlfx.fem.Constant( region, (0, -rho_ice * g), )


# set up parameter for simulation
p_water = 5e6
v_constant_in = 1.0 # velocity [m/day]
v_constant_out = v_constant_in #simulation is symmetric, so same velocity at outflow. This can be changed
eta_boundary = 1e13
T_end = 1 # XX days
dt_SI = 60*60      # XX seconds


# =============================================================================
# MPI AND READ MESH
# =============================================================================

dlfx.log.set_log_level(dlfx.log.LogLevel.WARNING)
comm = MPI.COMM_WORLD


if comm.rank == 0:
    print("Reading mesh ...")

sys.stdout.flush()

with dlfx.io.XDMFFile(comm, filename_mesh, 'r') as mesh_inp:
    region = mesh_inp.read_mesh()
    dim = region.topology.dim 
    fdim = dim-1
    region.topology.create_connectivity(fdim, dim)
    facet_tags = mesh_inp.read_meshtags(region, name='facet_tags')
    mesh_geometry = region.geometry.x
if comm.rank == 0:
    print("Mesh read.")
sys.stdout.flush()


# =============================================================================
# SIMULATION FUNCTION
# =============================================================================

def run_for_one_parameter_set(
        v_constant_in,
        v_constant_out, 
        p_water, 
        eta_boundary, 
        T_end, 
        dt_SI
        ):


    # ----------------------------------------------------------------------
    # Define FINITE-ELEMENT FUNCTION SPACES 
    # ----------------------------------------------------------------------

    # Define finite-element function spaces
    Se = element('CG', region.basix_cell(), 1) #for scalar quantities
    Ve = element('Lagrange', region.basix_cell(), 1, shape=(dim,)) #for vector-valued displacement space: u = (u_x, u_y)
    Te = element('DG', region.basix_cell(), 0, shape=(dim,dim)) #for tensor-valued output quantities
    deg_quad = 2
    QTe = quadrature_element(region.basix_cell(), value_shape=(dim, dim), scheme="default", degree=deg_quad) ## Quadrature space 

    # Create DOLFINx function spaces
    S = dlfx.fem.functionspace(region, Se)
    V = dlfx.fem.functionspace(region, Ve)
    V1 = dlfx.fem.functionspace(region, ("Lagrange", 1, (dim,)))
    QT = dlfx.fem.functionspace(region, QTe)
    T = dlfx.fem.functionspace(region, Te)


    # ----------------------------------------------------------------------
    # Define FUNCTIONS AND VARIABLES
    # ----------------------------------------------------------------------

    # u : current displacement 
    # u_old : displacement from previous time step 
    u = dlfx.fem.Function(V, name='u') 
    u_old = dlfx.fem.Function(V, name='u_old') 

    # Trial and test functions for the displacement equation. 
    du = ufl.TrialFunction(V) 
    ddu = ufl.TestFunction(V)

    # Viscous strain tensor
    # epsv : current viscous strain 
    # epsv_old : viscous strain from previous time step 
    epsv = dlfx.fem.Function(QT) 
    epsv_old = dlfx.fem.Function(QT, name='eps_vold') 

    # Temperature and viscosity fields.
    temperature = dlfx.fem.Function(S, name="temperature")
    eta = dlfx.fem.Function(S, name="viscosity")

    # Glen flow-rate factor A(T)
    Arate = dlfx.fem.Function(S, name="A(T)")


    # Fields used only for XDMF output
    XDMF_u = dlfx.fem.Function(V1, name="XDMF_u")
    XDMF_eps = dlfx.fem.Function(T, name='eps')
    XDMF_eps_v = dlfx.fem.Function(T, name='epsv')
    XDMF_sig_el =  dlfx.fem.Function(T, name='sig_el')
    XDMF_sig_v =  dlfx.fem.Function(T, name='sig_v')
    XDMF_strainrate = dlfx.fem.Function(T, name="strainrate")



    # Fields used for cavity/contact diagnostics 
    f_cavity_exact_int = dlfx.fem.Function(S, name='f_cavity_exact')
    f_cavity_int = dlfx.fem.Function(S, name='f_cavity')
    f_under_ground_int = dlfx.fem.Function(S, name='f_under_ground')
    lambda_ind = dlfx.fem.Function(S, name='lambda_ind')


    # ----------------------------------------------------------------------
    # MATERIAL PARAMETERS 
    # ----------------------------------------------------------------------

    # Elastic parameters
    E = dlfx.fem.Constant(region, dlfx.default_scalar_type(1e9))
    nu = dlfx.fem.Constant(region, dlfx.default_scalar_type(0.325)) 
    mu = E/(2*(1+nu))                    
    K = E / (3* (1 - 2 * nu) )
    
    # Viscous / Glen-flow parameters
    R = 8.31446261815324  # Gas constant [J/(mol K)]
    Q_plus = 115e3        # Activation energy above 263 K [J/mol]
    Q_minus = 60e3        # Activation energy below 263 K [J/mol]
    Astar = 3.5e-25       # Reference flow-rate factor for ice in Glen's flow law
    eps_min = 1e-15       # Small numerical regularisation of the effective strain rate
    n = 3                 # Glen exponent

    # ----------------------------------------------------------------------
    # OUTPUT FILE NAME 
    # ----------------------------------------------------------------------

    output_filename_with_parameters = (
        f"{output_filename}"
        f"_v{v_constant_in:.1f}"
        f"_p{p_water:.1e}"
        f"_T{T_end:.0f}"
        f"_eta{eta_boundary:.2e}"
    )
    


    # ----------------------------------------------------------------------
    # UNIT CONVERSION 
    # ----------------------------------------------------------------------
    
    # Internal time unit of the simulation is seconds.
    seconds_per_day = 24*60*60 

    v_constant_in_SI = v_constant_in/seconds_per_day
    v_constant_out_SI = v_constant_out/seconds_per_day

    T_end_SI = T_end * seconds_per_day


    # ----------------------------------------------------------------------
    # BOUNDARY CONDITIONS 
    # ----------------------------------------------------------------------

    x = ufl.SpatialCoordinate(region) # geometric coordinates

    if comm.rank == 0:
        print("Initialize BCs ...")
    sys.stdout.flush()


   # Find mesh tags for defining physical boundaries

    bedrock_tag = 1
    inflow_tag = 5
    outflow_tag = 3
    cavity = 2

    #Find degrees of freedom belonging to each boundary

    bedrock_y = dlfx.fem.locate_dofs_topological(V.sub(1), fdim, facet_tags.find(bedrock_tag))
    inflow_x = dlfx.fem.locate_dofs_topological(V.sub(0), fdim, facet_tags.find(inflow_tag)) 
    outflow_x = dlfx.fem.locate_dofs_topological(V.sub(0), fdim, facet_tags.find(outflow_tag))

    # Dirichlet boundary conditions for the displacement field u

    # Bedrock: vertical displacement is fixed
    Bedrock_y = dlfx.fem.dirichletbc(0.0, bedrock_y, V.sub(1)) 

    # Inflow and outflow displacement
    Inflow_x = dlfx.fem.dirichletbc(u_total_in, inflow_x, V.sub(0)) 
    Outflow_x = dlfx.fem.dirichletbc(u_total_out, outflow_x, V.sub(0)) 

    bcs = [Bedrock_y, Inflow_x, Outflow_x] 


    # ----------------------------------------------------------------------
    # CONSTITUTIVE RELATIONS 
    # ----------------------------------------------------------------------

    def temp_initial(x):
        """Initial temperature profile""" 

        y = -x[1]

        return 271.0 - 18.0 * (y**2) / ice_height**2


    def A_expr(T):
        """Arrhenius-type Glen flow law parameter for ice creep."""
        Q = ufl.conditional(ufl.lt(T, 263), Q_minus, Q_plus)
        return Astar * ufl.exp(-(Q/R)*((1/T)-(1/263)))
    
    def eta_expr(temp_field, epsvrate):
        """Effective viscosity based on the strain-rate norm."""
        eps_eff = ufl.sqrt(0.5 * ufl.inner(epsvrate, epsvrate) + eps_min**2)
        eta_eff = 0.5 * A_expr(temp_field)**(-1.0/n) * eps_eff**((1.0-n)/n)
        return eta_eff

    def eps(u):
        """Small-strain tensor from the displacement field."""
        return ufl.sym(ufl.grad(u))

    def sigma_v(u, eps_vold, dt, eta):
        """Viscous stress contribution."""
        return K * ufl.tr(eps(u)) * ufl.Identity(dim) + 2 * eta * eps_v_rate(eps_v(u, eps_vold, dt, eta), eps_vold, dt)

    def eps_v(u, eps_vold, dt, eta): 
        """ Implicit update of the viscous strain tensor"""
        epsilon = eps(u)
        alpha = (mu/eta) * dt
        eps_v_new = (1.0/(1.0+alpha)) * ufl.dev(eps_vold) \
                    + (alpha/(1.0+alpha)) * ufl.dev(epsilon) \
                    + 0.5 * ufl.tr(eps_vold) * ufl.Identity(2)
        return eps_v_new

    def eps_v_rate(eps_v, eps_vold, dt):
        """Viscous strain rate"""
        return (eps_v - eps_vold) / dt

    # ----------------------------------------------------------------------
    # LOCAL L2 PROJECTION 
    # ----------------------------------------------------------------------

    def local_project(func, funcspace):
        """ Project a UFL expression onto a finite-element function space. 
        """
        trial = ufl.TrialFunction(funcspace)
        test  = ufl.TestFunction(funcspace)   

        a = ufl.inner(trial, test)*ufl.dx
        L = ufl.inner(func, test)*ufl.dx

        projection_problem = dlfx.fem.petsc.LinearProblem(
            a,
            L,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
            petsc_options_prefix="projection_problem",
        )
        
        return projection_problem.solve()

    # ----------------------------------------------------------------------
    # INITIAL CONDITIONS 
    # ----------------------------------------------------------------------


        
    ds = ufl.Measure('ds', domain=region, subdomain_data=facet_tags, metadata={'quadrature_degree':deg_quad}) #ToDo replace


    t = dlfx.fem.Constant(region, dlfx.default_scalar_type(0.0))

    #everything is at rest
    u.x.array[:] = 0. 
    u_old.x.array[:] = 0.
    epsv.x.array[:] = 0.
    epsv_old.x.array[:] = 0.


    u.x.scatter_forward()
    u_old.x.scatter_forward()
    epsv.x.scatter_forward()
    epsv_old.x.scatter_forward()


    #Initialize temperature and viscosity fields
    temperature.interpolate(temp_initial)
    eta_initial_expr = 0.5 * A_expr(temperature)**(-1.0/n) * (1e-12)**((1.0-n)/n)
    eta.interpolate(dlfx.fem.Expression(eta_initial_expr, S.element.interpolation_points))


    # ----------------------------------------------------------------------
    # CAVITY CONTACT / PENALTY TERMS 
    # ----------------------------------------------------------------------

    traction = -p_water * ufl.FacetNormal(region)  # water pressure traction on the cavity boundary


    # Geometry-dependent cavity indicators (for defining penalty terms)
    cond_x = ufl.ge(x[0], ice_L - v_constant_in_SI * t)
    cond_y = ufl.ge(x[1]+u[1], 1e-4)
    f_cavity_exact = ufl.conditional(ufl.And(cond_x, cond_y), 1.0, 0.0)
    f_cavity = ufl.conditional(ufl.ge(x[0]+u[0], ice_L), 1.0, 0.0)
    f_under_step = ufl.conditional(ufl.lt(x[1]+u[1], step_height), 1.0, 0.0)
    f_under_ground = ufl.conditional(ufl.lt(x[1]+u[1], 0), 1.0, 0.0)

    def indicator_negative(expr):
        """Return 1 if a scalar expression is negative, otherwise 0."""
        return ufl.conditional(ufl.lt(expr, 0), 1.0, 0.0)



    # ----------------------------------------------------------------------
    # CONTACT / AUGMENTED-LAGRANGIAN VARIABLES 
    # ----------------------------------------------------------------------

    # Positive value means penetration below the ground.
    posY = -(x[1] + u[1]) 
    # Horizontal distance from the step.
    X_def = x[0] + u[0] - ice_L 

    # Penalty parameter.
    rho_base = 5.0e5           
    rho = dlfx.fem.Constant(region, dlfx.default_scalar_type(rho_base))

    # Penalty potential for penetration.
    Phi_pen = 0.5 * rho * posY**2  * f_cavity * f_under_ground * (ds(cavity) + ds(bedrock_tag))
    pen_contrib = ufl.derivative(Phi_pen, u, ddu)

    # Lagrange multiplier for the contact constraint.
    lambda_fun = dlfx.fem.Function(S, name="lambda")
    lambda_fun.x.array[:] = 0   # irgendein Wert
    lambda_fun.x.scatter_forward()  
    lagr_contrib = - lambda_fun * ddu[1]* f_cavity * f_under_ground  * (ds(cavity) + ds(bedrock_tag)) 

    penalty_under_ground = pen_contrib + lagr_contrib

    # Additional penalty preventing the ice from moving # too far before the step.
    penalty_before_step =  (
        +1*1e8 *X_def*f_under_step* ufl.dot(indicator_negative(x[0]+u[0]-ice_L), ddu[0])*(ds(cavity) + ds(bedrock_tag))) 



    # ----------------------------------------------------------------------
    # WEAK FORM AND LINEAR PROBLEM
    # ----------------------------------------------------------------------

    form = (
        ufl.inner(sigma_v(du, epsv_old, dt_SI, eta), eps(ddu)) * ufl.dx
        - ufl.inner(f, ddu) * ufl.dx
        - ufl.inner(traction, ddu) *  f_cavity_exact * (ds(cavity) + ds(bedrock_tag))
        + penalty_under_ground
        + penalty_before_step
    )



    a = ufl.lhs(form) 
    L = ufl.rhs(form) 

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="hydro_problem",
    ) 

    # ----------------------------------------------------------------------
    # COMPUTE CAVITY AREA
    # ----------------------------------------------------------------------



    def shoelace(x_y): 
        """ Calculate polygon area using the Shoelace formula. """
        x = x_y[:,0]
        y = x_y[:,1]

        S1 = np.sum(x*np.roll(y,-1))
        S2 = np.sum(y*np.roll(x,-1))

        area = .5*np.absolute(S1 - S2)

        return area


    def calculate_volume():
        """ Calculate cavity area from the displaced boundary points. """
        

        # Find cavity boundary degrees of freedom
        local_dofs = dlfx.fem.locate_dofs_topological(V, fdim, ft_new.find(cavity))
        u_coords = V.tabulate_dof_coordinates()
        cav_coords = u_coords[local_dofs][:,:2] + u.x.array.reshape(-1,2)[local_dofs]


        # Gather all cavity points on all MPI ranks.
        global_cav_coords = comm.allgather(cav_coords)
        global_cav_coords = np.concatenate(global_cav_coords, axis=0)

        # Initialize output quantities.
        A_original = None
        A_err = None
        y_neg_min = None
        A_physical = None
    

        # Detect points below y = 0 

        y_vals = global_cav_coords[:, 1]
        neg_mask = y_vals <= 0
    
 

        if np.any (y_vals < 0):
            neg_points = global_cav_coords[neg_mask]


            y_neg_min = np.min(neg_points[:, 1])


        # Estimate area caused by negative y-values 
            
        if np.any(neg_mask):
            neg_points = global_cav_coords[neg_mask]

            neg_points_sort = neg_points[neg_points[:,0].argsort()]

            x_min = neg_points_sort[0, 0]
            x_max = neg_points_sort[-1, 0]
            p_left  = np.array([x_min, 0.0])
            p_right = np.array([x_max, 0.0])

            polygon = np.vstack([
            neg_points_sort,     
            p_right,
            p_left
            ])

            A_err = shoelace(polygon)



        # Calculate physical cavity area 
        if comm.rank == 0:
            mask = global_cav_coords[:, 0] <= x_min
            filtered_coords = global_cav_coords[mask]

            filtered_coords_sort = filtered_coords[filtered_coords[:,0].argsort()]
            circ = np.vstack((np.array([  [ice_L, 0]]), filtered_coords_sort[:, :2]))
            circ = circ[circ[:,1] < 100] 

            A_physical = shoelace(circ)

            


            global_cav_coords[global_cav_coords<=0] = 1e-6 
            global_cav_coords_sort = global_cav_coords[global_cav_coords[:,0].argsort()]
            

            circ_original = np.vstack((np.array([  [ice_L, 0]]), global_cav_coords_sort[:, :2]))
            circ_original = circ_original[circ_original[:,1] < 100]  


            A_original = shoelace(circ_original)


        # All MPI ranks need the calculated areas
        A = comm.bcast(A_original, root=0)
        A_phy = comm.bcast(A_physical, root=0)

        return A, A_err, y_neg_min, A_phy



    # ----------------------------------------------------------------------
    # INITIALIZE TIME LOOP AND OUTPUT 
    # ----------------------------------------------------------------------
    j = 0


    # Create XDMF file and write the mesh once
    with dlfx.io.XDMFFile(region.comm, f"{output_filename_with_parameters}.xdmf", "w") as xdmf:
        xdmf.write_mesh(region)

    nsteps = int(np.floor(T_end_SI / dt_SI))

    # Arrays storing variables over time.
    strainrate = dlfx.fem.Function(QT)
    strainrate_array = np.empty(len(np.arange(0, nsteps + 1)))
    eta_array = np.empty(len(np.arange(0, nsteps + 1)))
    eta_array_min = np.empty(len(np.arange(0, nsteps + 1)))
    strainrate_array_min = np.empty(len(np.arange(0, nsteps + 1)))


    A_arr = []
    A_err_arr = []
    y_neg_min_arr = []
    A_phy_arr = []

    contact_has_closed_once = False


    # ---------------------------------------------------------------------- 
    # MAIN TIME LOOP 
    # ----------------------------------------------------------------------

    for istep in range(0, nsteps + 1):
        t.value = istep * dt_SI

        u_total_in.value = (v_constant_in_SI) * t.value
        u_total_out.value = (v_constant_out_SI) * t.value

        t_days = t.value/(60*60*24)
        print(f"[Rank {comm.rank}] Solving at t = {t_days:.2f}d")
        sys.stdout.flush()


        # ---------------------------------------------------------- 
        #  Augmented-Lagrangian contact iteration for all time steps except the first one
        # ----------------------------------------------------------
        if istep > 0:


            AL_iters = 10
            for k_al in range(AL_iters):
                

                uh = problem.solve() 
                u.x.array[:] = uh.x.array[:]
                u.x.scatter_forward()

                # Update lambda using current displacement u
                dofs_cavity = dlfx.fem.locate_dofs_topological(S, fdim, facet_tags.find(cavity)).astype(np.int32)
                dofs_bedrock = dlfx.fem.locate_dofs_topological(S, fdim, facet_tags.find(bedrock_tag)).astype(np.int32)
                contact_dofs = np.unique(np.concatenate((dofs_cavity, dofs_bedrock)))

                coords_all = S.tabulate_dof_coordinates()
                gdim = region.geometry.x.shape[1] 
                coords_all = coords_all.reshape(-1, gdim)
                valid_mask = contact_dofs < coords_all.shape[0]
                contact_local = contact_dofs[valid_mask]
                coords_contact = coords_all[contact_local]
                nodal_u_all = u.x.array.reshape(-1, dim)
                nodal_u = nodal_u_all[contact_local]

                gap = -(coords_contact[:, 1] + nodal_u[:, 1])   
                pos_gap = np.maximum(gap, -1e-9)

                lambda_local = lambda_fun.x.array.copy()
                

                local_max_pos = float(np.max(pos_gap)) if pos_gap.size > 0 else 0.0
                global_max_pos = comm.allreduce(local_max_pos, MPI.MAX)


                # Check contact convergence

                if global_max_pos < 1e-2: # toleance for convergence of contact iteration = 2cm

                    open_far = gap < -1e-9
                    lambda_local[contact_dofs[open_far]] *= 0.1
                    lambda_fun.x.array[:] = lambda_local
                    lambda_fun.x.scatter_forward()

                    break



                #find if contact has closed
                if not contact_has_closed_once:
                    if global_max_pos > 1e-4:
                        contact_has_closed_once = True
                        if comm.rank == 0:
                            print("Contact has closed.")


                # Augmented-Lagrangian update
                lambda_local[contact_dofs] = np.maximum(0.0, lambda_local[contact_dofs] + float(1e7) * pos_gap)

                # Reduce multiplier on open boundaries.
                open_far = gap < 1e-9
                lambda_local[contact_dofs[open_far]] *= 0.1

                lambda_fun.x.array[:] = lambda_local
                lambda_fun.x.scatter_forward()


                
        # -------------------------------------------------------------- 
        # Initial time step: no contact iteration 
        # --------------------------------------------------------------
               
        else:
            uh = problem.solve()
            u.x.array[:] = uh.x.array[:]
            u.x.scatter_forward()


        # ----------------------------------------------------------------------
        # CONTACT DIAGNOSTICS 
        # ----------------------------------------------------------------------

        local_max = np.max(lambda_fun.x.array)
        global_max = comm.allreduce(local_max, op=MPI.MAX)


        print(f"[Rank {comm.rank}] Solved at t = {t_days:.2f}d")
        sys.stdout.flush()

        # ----------------------------------------------------------------------
        # UPDATE AND PROJECT VARIABLES FOR OUTPUT
        # ----------------------------------------------------------------------

        # Viscous strain and strain rate
        epsv.interpolate(dlfx.fem.Expression(eps_v(u,epsv_old,dt_SI, eta), QT.element.interpolation_points))
        strainrate.interpolate(dlfx.fem.Expression(eps_v_rate(epsv, epsv_old, dt_SI), QT.element.interpolation_points))
        
        strainrate_print = local_project(eps_v_rate(epsv, epsv_old, dt_SI), T)
        eta_ = local_project(eta_expr(temperature, eps_v_rate(epsv, epsv_old, dt_SI)), S)

        # Glen flow-rate factor
        Arate.interpolate(dlfx.fem.Expression(A_expr(temperature), S.element.interpolation_points))

        # Cavity/contact indicators and Lagrange multiplier
        f_cavity_exact_int.interpolate(dlfx.fem.Expression(f_cavity_exact, S.element.interpolation_points))
        f_cavity_int.interpolate(dlfx.fem.Expression(f_cavity, S.element.interpolation_points))
        f_under_ground_int.interpolate(dlfx.fem.Expression(f_under_ground, S.element.interpolation_points))
        lambda_ind.interpolate(dlfx.fem.Expression(lambda_fun, S.element.interpolation_points))


        # Global maximum and minimum values
        strainrate_max = comm.allreduce(np.max(strainrate_print.x.array), op=MPI.MAX)
        eta_max = comm.allreduce(np.max(eta_.x.array), op=MPI.MAX)
        strainrate_array[j] = strainrate_max
        eta_array[j] = eta_max
        strainrate_min = comm.allreduce(np.max(strainrate_print.x.array), op=MPI.MIN)
        eta_min = comm.allreduce(np.max(eta_.x.array), op=MPI.MIN)
        strainrate_array_min[j] = strainrate_min
        eta_array_min[j] = eta_min


        # ----------------------------------------------------------------------
        # UPDATE DYNAMIC BOUNDARY TAGS 
        # ----------------------------------------------------------------------


        tol = 1e-6  # Tolerance when identifying geometric boundaries

        def in_cavity(x):
            """ Identify facets currently belonging to the cavity. """
            return np.logical_and(x[0] >= ice_L - float(v_constant_in_SI * t.value),x[0] <= ice_L + float(v_constant_in_SI * t.value+ step_length+100))
   

        def bedrock(x): 
            """Identify the moving bedrock boundary."""
            shift = float(v_constant_in_SI * t.value)

            cond_right = np.logical_and( x[0] > (ice_L + step_length  - tol), np.isclose(x[1], 0.0, atol=tol))

            cond_left_step = np.logical_and( x[0] < (ice_L + tol - shift), np.isclose(x[1], step_height, atol=tol))

            return np.logical_or(cond_right, cond_left_step)


        def inflow(x):
            """Identify inflow boundary."""
            return np.isclose(x[0], 0.0, atol=tol)
        
        def outflow(x):
            """Identify outflow boundary."""
            return np.isclose(x[0], ice_length, atol=tol)



        cavity_ent = dlfx.mesh.locate_entities_boundary(region, fdim, in_cavity)
        bedrock_ent = dlfx.mesh.locate_entities_boundary(region, fdim, bedrock)
        inflow_ent = dlfx.mesh.locate_entities_boundary(region, fdim, inflow) 
        outflow_ent = dlfx.mesh.locate_entities_boundary(region, fdim, outflow)



        marked_facets = np.hstack([cavity_ent, bedrock_ent, inflow_ent, outflow_ent])
        marked_values = np.hstack([np.full_like(cavity_ent, cavity), np.full_like(bedrock_ent, bedrock_tag), np.full_like(inflow_ent, inflow_tag), np.full_like(outflow_ent, outflow_tag)])

        sorted_facets = np.argsort(marked_facets)

        ft_new = dlfx.mesh.meshtags(region, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

        # ----------------------------------------------------------------------
        # UPDATE BOUNDARY CONDITIONS 
        # ----------------------------------------------------------------------


        bedrock_y = dlfx.fem.locate_dofs_topological(V.sub(1), fdim, ft_new.find(bedrock_tag))

        inflow_x = dlfx.fem.locate_dofs_topological(V.sub(0), fdim, ft_new.find(inflow_tag)) 
        outflow_x = dlfx.fem.locate_dofs_topological(V.sub(0), fdim, ft_new.find(outflow_tag))

        
        Bedrock_y = dlfx.fem.dirichletbc(0.0, bedrock_y, V.sub(1)) 
        Inflow_x = dlfx.fem.dirichletbc(u_total_in, inflow_x, V.sub(0)) 
        Outflow_x = dlfx.fem.dirichletbc(u_total_out, outflow_x, V.sub(0)) 


        bcs = [Bedrock_y, Inflow_x, Outflow_x] 

        # Update integration measure using the new boundary tags
        ds = ufl.Measure('ds', domain=region, subdomain_data=ft_new, metadata={'quadrature_degree':deg_quad})

        # Update boundary conditions used by the solver.
        problem.bcs = bcs 

        # ----------------------------------------------------------------------
        # PREPARE DATA FOR OUTPUT 
        # ----------------------------------------------------------------------


        XDMF_u.interpolate(u)

        eps_v_ = local_project(eps_v(u, epsv_old, dt_SI, eta), T)
        XDMF_eps_v.x.array[:] = eps_v_.x.array[:]
        XDMF_eps_v.x.scatter_forward()

        XDMF_eps.interpolate(dlfx.fem.Expression(eps(u), T.element.interpolation_points)) 

        sig_v = local_project(sigma_v(u, epsv_old, dt_SI, eta), T)
        XDMF_sig_v.x.array[:] = sig_v.x.array[:]
        XDMF_sig_v.x.scatter_forward()

        XDMF_strainrate.x.array[:] = strainrate_print.x.array[:]

        eta.x.array[:] = np.maximum(eta_.x.array, eta_boundary)
        eta.x.scatter_forward()

        # ----------------------------------------------------------------------
        # CAVITY AREA  
        # ----------------------------------------------------------------------


        A_abs,A_err, y_neg_min, A_phy= calculate_volume()

        A_arr.append(A_abs)
        y_neg_min_arr.append(y_neg_min)
        A_phy_arr.append(A_phy)
        A_err_arr.append(A_err)

        A_abs_array = np.array(A_arr)
        A_err_abs_array = np.array(A_err_arr)
        y_neg_min_array = np.array(y_neg_min_arr)
        A_phy_array = np.array(A_phy_arr)



        


        #saving maximal penetration depth for each time step
        penetrartion_depth_max = []
        penetrartion_depth_max.append(y_neg_min)
        penetrartion_depth_max_array = np.array(penetrartion_depth_max)

        pen_max = np.min(penetrartion_depth_max_array)

        if istep == nsteps and comm.rank == 0:
            print(f"Maximal penetration depth over the simulation: {pen_max:.4f} m")
            sys.stdout.flush()




        

        # ----------------------------------------------------------------------
        # WRITE XDMF OUTPUT 
        # ----------------------------------------------------------------------


        if istep in np.arange(0, nsteps+1, 1):
            with dlfx.io.XDMFFile(region.comm, f"{output_filename_with_parameters}.xdmf", "a") as xdmf:
                try:
                    """ Write XDMF output for the current time step, variables of interest can be added or removed as needed. """
                    xdmf.write_function(XDMF_u, t.value)
                    xdmf.write_function(XDMF_eps, t.value)
                    #xdmf.write_function(XDMF_eps_v, t.value)
                    #xdmf.write_function(temperature, t.value)
                    #xdmf.write_function(eta, t.value)
                    #xdmf.write_function(f_cavity_int, t.value)
                    #xdmf.write_function(f_cavity_exact_int, t.value)
                    #xdmf.write_function(f_under_ground_int, t.value)
                    xdmf.write_function(lambda_ind, t.value)
                    #xdmf.write_function(XDMF_strainrate, t.value)
                    #xdmf.write_function(XDMF_sig_v, t.value)
                    #xdmf.write_function(XDMF_sig_el, t.value)

                    print(f"[Rank {comm.rank}] Wrote XDMF for t = {t_days:.2f}d")
                    sys.stdout.flush()
                except RuntimeError as e:
                    print(f"[Rank {comm.rank}] Error while writing XDMF:", e)
                    sys.stdout.flush()


        # ----------------------------------------------------------------------
        # PREPARE FOR NEXT TIME STEP 
        # ----------------------------------------------------------------------
        epsv_old.x.array[:] = epsv.x.array[:]
        epsv_old.x.scatter_forward()
        j += 1

        comm.barrier()
        eta.x.scatter_forward()
        epsv.x.scatter_forward()
        epsv_old.x.scatter_forward()
        strainrate.x.scatter_forward()


        # ----------------------------------------------------------------------
        # SAVE TIME-SERIES DATA 
        # ----------------------------------------------------------------------


        if comm.rank == 0:
            print("Saving data ...")
        sys.stdout.flush()
        if comm.rank == 0:
            time_array = np.arange(0, nsteps+1)
            np.savez(f'{output_filename_with_parameters}', area=A_abs_array, area_err = A_err_abs_array, y_min = y_neg_min_array, A_phy = A_phy_array , time=time_array)

    ##########################


runtime_step = time.time()

run_for_one_parameter_set(v_constant_in, v_constant_out, p_water, eta_boundary, T_end, dt_SI)
runtime_end = time.time()
total_runtime = (runtime_end - runtime_start)/60


if comm.rank == 0:
    print(f"Finished with after t={(runtime_end - runtime_step):.1f}s; simulation running for {total_runtime:.2f}min.")

if comm.rank == 0:
    print('Ich habe fertig, Kollege!')
    sys.stdout.flush()