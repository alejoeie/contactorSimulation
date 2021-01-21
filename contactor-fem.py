"""
    Se modela un contactor magnetico en vivo considerando geometria y analisis 
    de los modelos magneticos que lo definen. Se busca un paper de un contactor
    que da las dimensiones, la corriente nominal y las tensiones del mismo.
    
    Para correrlo, debe introducir:
    python3 contactor-fem.py <distancia>

    Donde <distancia> es un valor float entre 0.0 y 0.8

    Ej: 
    python3 contactor-fem.py 0.4

"""


# Se importan las librerias necesarias
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import dolfin.common.plotting as mshplot
import sys

# define distance from user

dist = float(sys.argv[1])

def sum_points(x1, x2):
    ''' Returns vector sum of two points. '''

    return (x1[0] + x2[0], x1[1] + x2[1])

def draw_path(rel_coords, origin=(0, 0)):
    ''' Transforms a path with relative coordinates
        into one with absolute coordinates. '''

    # Transform to absolute coordinates
    last_coord = rel_coords[0]
    abs_coords = [last_coord]
    for coord in rel_coords[-len(rel_coords)+1::]:
        next_coord = sum_points(last_coord, coord)
        abs_coords.append(next_coord)
        last_coord = next_coord

    return abs_coords

def shift_origin(coords, origin=(0,0)):
    ''' Modifies path with absolute coordinates so that
        the first coordinate matches a given point (origin)
        and distances are preserved. '''
    
    offset = (origin[0] - coords[0][0], origin[1] - coords[0][1])

    return [sum_points(offset, coord) for coord in coords]

# Define dimensions
thickness = 0.8
la = 1.6
lb = 1.05
center_thickness = 1.1
ring_depth = 0.15
ring_width = 0.1
ring_sep = 0.1
center_gap = 0.04

# Draw fixed core starting at bottom-left corner and going couter-clockwise
rel_vertices=[(0, 0), 
              (2*thickness + 2*lb + center_thickness, 0),
              (0, thickness + la),
              (-ring_sep, 0),
              (0, -ring_depth),
              (-ring_width, 0),
              (0, ring_depth),
              (-thickness + 2*ring_width + 2*ring_sep, 0),
              (0, -ring_depth),
              (-ring_width, 0),
              (0, ring_depth),
              (-ring_sep, 0),
              (0, -la),
              (-lb, 0),
              (0, la - center_gap),
              (-center_thickness, 0),
              (0, -la + center_gap),
              (-lb, 0),
              (0, la),
              (-ring_sep, 0),
              (0, -ring_depth),
              (-ring_width, 0),
              (0, ring_depth),
              (-thickness + 2*ring_width + 2*ring_sep, 0),
              (0, -ring_depth),
              (-ring_width, 0),
              (0, ring_depth),
              (-ring_sep, 0)]

abs_vertices = draw_path(rel_vertices)
shifted_vertices = shift_origin(abs_vertices, origin=(1, 3.4 - dist - la - thickness))
points = [Point(v) for v in shifted_vertices]
print(shifted_vertices)


upper_vertices = [Point(5.8, 5.8),
                Point(1, 5.8),
                Point(1,3.4),
                Point(1.8, 3.4),
                Point(1.8, 5),
                Point(2.85, 5),
                Point(2.85, 3.4),
                Point(3.95, 3.4),
                Point(3.95, 5),
                Point(5, 5),
                Point(5, 3.4),
                Point(5.8, 3.4)]

# upper_vertices = Polygon(points)

domain = Circle(Point(1+thickness+lb+0.5*center_thickness,3.4-0.5*dist),10) 

# Define wire geometry
winding_sep = 0.1*lb
winding = [(0, 0),
           (lb - 2*winding_sep, 0),
           (0, la - 2*winding_sep),
           (-lb + 2*winding_sep, 0)]


left_winding = shift_origin(draw_path(winding), 
                            origin=(1 + thickness + winding_sep, 3.4 - dist - la + winding_sep))
right_winding = shift_origin(draw_path(winding), 
                             origin=(1 + thickness + winding_sep + lb + center_thickness, 3.4 - dist - la + winding_sep))

left_points = [Point(v) for v in left_winding]
right_points = [Point(v) for v in right_winding]

wire1 = Polygon(left_points)
wire2 = Polygon(right_points)


circuit = Polygon(points) + Polygon(upper_vertices) + wire1 + wire2
domain.set_subdomain(1, circuit)
domain.set_subdomain(2, wire1)
domain.set_subdomain(3, wire2)

msh = generate_mesh(circuit, 24)
plot(msh)
plt.axis([0.5, 6.3, 0.5, 6.3])
plt.show()
# Generate and plot mesh
mesh = generate_mesh(domain, 32)
# plot(mesh)
# plt.axis([0.5, 6.3, 0.5, 6.3])
# plt.show()


# exit()
# Definir el espacio de la funcion
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
bc = DirichletBC(V, Constant(0), 'on_boundary')

# Define subdomain markers and integration measure
markers = MeshFunction('size_t', mesh, 2, mesh.domains())
dx = Measure('dx', domain=mesh, subdomain_data=markers)

# Define current densities
J_N = Constant(2*1100e1/(1.4*0.7))
J_S = Constant(-2*1100e1/(1.4*0.7))

# Define magnetic permeability
class Permeability(UserExpression):
    def __init__(self, markers, **kwargs):
        super(Permeability, self).__init__(**kwargs)
        self.markers = markers

    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            # Air
            values[0] = 4*pi*1e-7
        elif self.markers[cell.index] == 1:
            # Iron
            values[0] = 6.3e-3
        else:
            # Copper
            values[0] = 1.26e-6

    def value_shape(self):
	    return ()

mu1 = Permeability(markers, degree=1)


# Define variational problem
A_z = TrialFunction(V)
v = TestFunction(V)
a = (1 / mu1)*dot(grad(A_z), grad(v))*dx
L_N = J_N*v*dx(2)
L_S = J_S*v*dx(3) 
L = L_N + L_S

# Solve variational problem for Az
A_z = Function(V)
solve(a == L, A_z, bc)

# Compute magnetic field as B == curl A
W = VectorFunctionSpace(mesh, 'P', 1)
B = project(as_vector((A_z.dx(1), -A_z.dx(0))), W)

# Initialize axes for plotting solution
ax = plt.axes()


# Draw mesh and solution
mshplot.plot(mesh, linewidth=0.2)
c = plot(B)
plt.axis([0.5, 6.3, 0.5, 6.3])
plt.title('Campo magnético (T)')
plt.colorbar(c)
plt.show()


# Compute magnitude of B
Bx =  A_z.dx(1)
By = -A_z.dx(0)
B_abs = np.power(Bx**2 + By**2, 0.5)

# Define new function space as Discontinuous Galerkin
abs_B = FunctionSpace(mesh, 'DG', 0)
f = B_abs

# Define new variational problem
w_h = TrialFunction(abs_B)
v = TestFunction(abs_B)
a = w_h*v*dx
L = f*v*dx

# Solve additional problem for abs_B
w_h = Function(abs_B)
solve(a == L, w_h)

# Plot magnitude of B
mshplot.plot(mesh, linewidth=0.2)
c = plot(w_h)
plt.axis([0.5, 6.3, 0.5, 6.3])
plt.title('Magnitud de campo magnético (T)')
plt.colorbar(c)
plt.show()





