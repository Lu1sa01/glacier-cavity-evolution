import numpy as np
import dolfinx as dlfx
import gmsh
from mpi4py import MPI
import time


# =============================================================================
# Configuration
# =============================================================================

# Prefix for the generated mesh files.
# The geometry parameters and mesh resolutions are appended to this prefix to create a unique filename for each mesh configuration.


# Local execution path.
mesh_name_prefix = "/root/share/Hydro/output/Mesh/testing/coarse"

# Cluster execution path.
# Uncomment the following line when running on the cluster.
#TU_ID = ""
# mesh_name_prefix = (
#     f"/home/{TU_ID}/NUIW_Simulations/Hydro/output/Mesh/testing/coarse"
# )

# Spatial dimension of the computational domain.
gdim = 2


# =============================================================================
# Mesh generation
# =============================================================================

def power_mesh_creation_function():
    """
    Configure the geometry and mesh resolution for the power-law mesh.

    Defines the cavity geometry, ice dimensions, and characteristic mesh sizes.
    The configured parameters are passed to create_mesh() to generate the
    computational mesh.

    Geometry dimensions are given in meters.
    """

    # Geometry parameters

    step_height = 1
    step_length = 5 * step_height

    ice_height = 1000  # Total height of the ice domain [m]
    ice_length = 2000  # Total horizontal length of the domain [m]

    # Mesh resolution parameters

    h_size_around_cavity_fine = 0.1  # Target element size near the cavity [m]
    h_size_around_cavity = 1   # Target element size near the cavity [m]
    h_size_around_top = 10    # Target element size near the upper boundary [m]
    h_size_bulk = 500           # Target element size in the bulk domain [m]

    print(
        f"Creating mesh with "
        f"h_cavity={h_size_around_cavity:.4f}, "
        f"h_bulk={h_size_bulk}, "
        f"h_top={h_size_around_top}..."
    )

    # Generate the mesh using the specified geometry and resolution.
    create_mesh(
        step_height,
        step_length,
        ice_height,
        ice_length,
        h_size_around_cavity,
        h_size_around_cavity_fine,
        h_size_bulk,
        h_size_around_top,
    )


def create_mesh(
    step_height,
    step_length,
    ice_height,
    ice_length,
    h_size_around_cavity,
    h_size_around_cavity_fine,
    h_size_bulk,
    h_size_around_top,
):
    """
    Construct and generate a two-dimensional mesh of the ice domain.

    The geometry consists of a rectangular ice domain with an elliptical
    ice-water interface above a bedrock cavity. Local mesh refinement is
    applied near the cavity and the upper and lower boundaries.

    Parameters
    ----------
    step_height : float
        Height of the bedrock step defining the cavity geometry [m].

    step_length : float
        Horizontal length associated with the cavity geometry [m].

    ice_height : float
        Total height of the ice domain [m].

    ice_length : float
        Total horizontal length of the computational domain [m].

    h_size_around_cavity_fine : float
        Target element size in the ultra-fine mesh region near the cavity [m].

    h_size_around_cavity : float
        Target element size near the cavity [m].

    h_size_bulk : float
        Target element size in the bulk domain [m].

    h_size_around_top : float
        Target element size near the upper boundary [m].

    """

    start_time = time.time()

    # Create a descriptive filename containing the geometry parameters and the target mesh resolutions.
    mesh_name = (
        f"{mesh_name_prefix}"
        f"_steph_{step_height}"
        f"_stepl_{step_length}"
        f"_iceh_{ice_height}"
        f"_icel_{ice_length}"
        f"_h_cavity_{h_size_around_cavity:.4f}"
        f"_h_bulk_{h_size_bulk}"
        f"_h_top_{h_size_around_top}"
    )

    # Initialize the Gmsh geometry model

    gmsh.option.setNumber("General.NumThreads", 4)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.clear()
    gmsh.model.add(mesh_name)
    gdim = 2

    # Geometry definition
    ice_L = (ice_length - step_length) / 2 # Horizontal distance between the cavity and the left and right edges.


    # Mesh refinement parameters

    dist_min = 50 * step_height
    dist_max = 0.5 * ice_height

    dist_top_min = 20
    dist_top_max = 100

    # Target element sizes.


    # Define geometric points (points are numbered counterclockwise around the outer boundary)

    point1 = gmsh.model.occ.add_point(0, step_height, 0) 
    point2 = gmsh.model.occ.add_point(ice_L, step_height, 0) 
    point3 = gmsh.model.occ.add_point(ice_L, 0, 0) #point for ellipse
    point4 = gmsh.model.occ.add_point(ice_L + step_length, 0, 0)
    point5 = gmsh.model.occ.add_point(2*ice_L + step_length, 0, 0)
    point6 = gmsh.model.occ.add_point(2*ice_L + step_length, ice_height, 0)  
    point7 = gmsh.model.occ.add_point(0, ice_height, 0)  
    point8 = gmsh.model.occ.add_point(ice_L + 1/2*step_length, 0, 0) #point on the main axis of the ellipse


    # Define boundary curves


    line1 = gmsh.model.occ.add_line(point1, point2) # Left section of the lower boundary.
    line2 = gmsh.model.occ.addEllipseArc(point2, point3, point8, point4) # Elliptical ice-water interface above the cavity.
    line3 = gmsh.model.occ.add_line(point4, point5) # Right section of the lower boundary.
    line4 = gmsh.model.occ.add_line(point5, point6) # Right vertical boundary.
    line5 = gmsh.model.occ.add_line(point6, point7) # Upper horizontal boundary.
    line6 = gmsh.model.occ.add_line(point7, point1) # Left vertical boundary.

    loop = gmsh.model.occ.add_curve_loop([line1, line2, line3, line4, line5, line6])
    area = gmsh.model.occ.add_plane_surface([loop])

    gmsh.model.occ.synchronize()

    # Define physical groups

    
    gmsh.model.addPhysicalGroup(1, [line1, line3], name="boundary_ice_bed", tag = 1)  # boundary between ice and bed
    gmsh.model.addPhysicalGroup(1, [line2], name="boundary_ice_water", tag = 2)  # boundary between ice and water
    gmsh.model.addPhysicalGroup(1, [line4], name="right_edge_v_out", tag = 3)  # right edge
    gmsh.model.addPhysicalGroup(1, [line5], name="top_edge", tag = 4)  # top edge
    gmsh.model.addPhysicalGroup(1, [line6], name="left_edge_v_in", tag = 5)  # left edge
    #gmsh.model.addPhysicalGroup(1, [line7], name="vertical_step_line", tag = 6)  # vertical step line
    gmsh.model.addPhysicalGroup(2, [area], name="area", tag = 1)  # area
    #gmsh.model.addPhysicalGroup(2, [areatwo], name="volumen_cavity", tag = 2)  #volume cavity



    # -------------------------------------------------------------------------
    # Define the background mesh size fields
    # -------------------------------------------------------------------------

    # Field 1 and Field 2: refinement near the cavity

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [line2])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_size_around_cavity)  # fine mesh size near the lower boundary
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_size_bulk)  # coarse mesh size above
    gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)  # area where fine mesh is applied
    gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)  # area where mesh becomes coarser


    # Field 3 and Field 4: refinement near the upper boundary
    gmsh.model.mesh.field.add("MathEval", 3)
    gmsh.model.mesh.field.setString(3, "F", f"1000 - y") 
    gmsh.model.mesh.field.add("Threshold", 4)  
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", h_size_around_top)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", h_size_bulk)
    gmsh.model.mesh.field.setNumber(4, "DistMin", dist_top_min)
    gmsh.model.mesh.field.setNumber(4, "DistMax", dist_top_max)



    # Field 6 and Field 7: refinement near the lower boundary

    gmsh.model.mesh.field.add("MathEval", 6)
    gmsh.model.mesh.field.setString(6, "F", f"y")
    gmsh.model.mesh.field.add("Threshold", 7)  
    gmsh.model.mesh.field.setNumber(7, "InField", 6)
    gmsh.model.mesh.field.setNumber(7, "SizeMin", h_size_around_top)
    gmsh.model.mesh.field.setNumber(7, "SizeMax", h_size_bulk)
    gmsh.model.mesh.field.setNumber(7, "DistMin", dist_top_min)
    gmsh.model.mesh.field.setNumber(7, "DistMax", dist_top_max)

    # Field 10: localized ultra-fine mesh refinement

    gmsh.model.mesh.field.add("Box", 10)
    gmsh.model.mesh.field.setNumber(10, "VIn", h_size_around_cavity_fine)
    gmsh.model.mesh.field.setNumber(10, "VOut", h_size_around_cavity)
    gmsh.model.mesh.field.setNumber(10, "XMin", ice_L-50)
    gmsh.model.mesh.field.setNumber(10, "XMax", ice_L)
    gmsh.model.mesh.field.setNumber(10, "YMin", step_height)
    gmsh.model.mesh.field.setNumber(10, "YMax", step_height+0.02)

    # -------------------------------------------------------------------------
    # Field 20 and Field 21: ultra-fine refinement near the interface
    # -------------------------------------------------------------------------
    gmsh.model.mesh.field.add("Distance", 20)
    gmsh.model.mesh.field.setNumbers(20, "CurvesList", [line2])
    gmsh.model.mesh.field.setNumber(20, "Sampling", 150)

    gmsh.model.mesh.field.add("Threshold", 21)
    gmsh.model.mesh.field.setNumber(21, "InField", 20)
    gmsh.model.mesh.field.setNumber(21, "SizeMin", h_size_around_cavity_fine)
    gmsh.model.mesh.field.setNumber(21, "SizeMax", h_size_bulk)
    gmsh.model.mesh.field.setNumber(21, "DistMin", 0.02)
    gmsh.model.mesh.field.setNumber(21, "DistMax", 0.04)
    

    # Combine mesh size fields

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4, 7, 10, 21])  
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # Generate the mesh

    gmsh.model.mesh.generate(dim=gdim)

    # Convert the Gmsh mesh to DOLFINx

    mesh_data = dlfx.io.gmsh.model_to_mesh(
    gmsh.model,
    MPI.COMM_WORLD,
    rank=0,
    gdim=gdim
)

    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    #Abspeicern als XDMF
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)



    # Export the mesh and physical tags

    # Write the mesh and its associated physical tags to an XDMF file.
    # The file can subsequently be read by DOLFINx for simulation purposes.
    with dlfx.io.XDMFFile(MPI.COMM_WORLD, mesh_name+'.xdmf', 'w') as xdmfout:
        xdmfout.write_mesh(mesh)
        xdmfout.write_meshtags(cell_tags, mesh.geometry) 
        xdmfout.write_meshtags(facet_tags, mesh.geometry) 

    # Report mesh generation statistics

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    elapsed = time.time() - start_time
    print(f"Done with mesh {mesh_name}")
    print(f"Mesh {mesh_name}: {num_cells} elements, generated in {elapsed:.2f} seconds")



# Main execution

gmsh.initialize()

try:
    power_mesh_creation_function()

finally:
    gmsh.finalize()