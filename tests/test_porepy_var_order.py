import porepy as pp
import numpy as np

m = pp.meshing.tensor_grid([2,2])
m.compute_geometry()
mdg = pp.MixedDimensionalGrid()
mdg.add_subdomains(m)
eq = pp.ad.EquationSystem(mdg)

v = pp.ad.Variable(mdg.subdomains(), 2, "u")
eq.variables.append(v)
# Set u = [0, 1, 0, 1, 0, 1, 0, 1] meaning [x0,y0,x1,y1,x2,y2,x3,y3]
vals = np.array([0, 1, 0, 1, 0, 1, 0, 1])
pp.set_solution_values("u", vals, mdg.subdomain_data(m), iterate_index=0)

val = v.evaluate(eq).val
print("Variable shape:", val.shape)
print("Variable vals:", val)
print("reshaped F:\n", val.reshape(2, -1, order='F'))
print("reshaped C:\n", val.reshape(2, -1, order='C'))
