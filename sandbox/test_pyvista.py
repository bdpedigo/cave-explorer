# %%
import pyvista as pv
from pyvista import examples

pv.set_jupyter_backend("client")

mesh = examples.download_bunny_coarse()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, color="white")
pl.add_points(mesh.points, color="red", point_size=2)
pl.camera_position = [(0.02, 0.30, 0.73), (0.02, 0.03, -0.022), (-0.03, 0.94, -0.34)]
pl.show()

# %%
mesh.point_data

# %%
import numpy as np

points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
lines = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]], dtype=np.int64)

# Create the PolyData
mesh = pv.PolyData(points, lines=lines)

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, color="black", line_width=5)
pl.add_points(mesh.points, color="red", point_size=10)
pl.camera_position = [(0.02, 0.30, 0.73), (0.02, 0.03, -0.022), (-0.03, 0.94, -0.34)]
pl.show()

# %%
import numpy as np
import pyvista as pv

points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
poly = pv.lines_from_points()
poly.plot(line_width=5)
