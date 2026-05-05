import porepy as pp
import numpy as np

m = pp.meshing.cart_grid([2,2], [1,1])
m.compute_geometry()
print("Centers:\n", m.cell_centers)

bc = pp.BoundaryConditionVectorial(m, m.get_all_boundary_faces(), "dir")
bc_values = np.zeros((2, m.num_faces))
bc_values[0, :] = 1.0 # set ux to 1
bc_values[1, :] = 2.0 # set uy to 2
print("bc_values ravel('F'):", bc_values.ravel("F"))
print("bc_values ravel('C'):", bc_values.ravel("C"))
