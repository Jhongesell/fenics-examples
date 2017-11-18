from fenics import *
import time

T = 2.5
# final time
num_steps = 1000
# number of time steps
dt = T / num_steps # time step size

# Create mesh
nx = ny = 50
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

#define function space
V = FunctionSpace(mesh, "P", 2)

# Tag boundaries
tol = 1e-14

class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, tol)

class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1, tol)

class BoundaryY0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, tol)

class BoundaryY1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1, tol)

# Mark boundaries
boundary_markers = FacetFunction('size_t', mesh)
boundary_markers.set_all(9999)
bx0 = BoundaryX0()
bx1 = BoundaryX1()
by0 = BoundaryY0()
by1 = BoundaryY1()
bx0.mark(boundary_markers, 0) # tag 0 to x = 0
bx1.mark(boundary_markers, 1) # tag 1 to x = 1
by0.mark(boundary_markers, 2) # tag 2 to y = 0
by1.mark(boundary_markers, 3) # tag 3 to y = 1

# Redefine boundary integration measure
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Define boundaries
# denics define natural Neumann boundaries by default
# so we dont need to specify right and left boundaries
#top   = "near(x[1], 1)"
bot   = "near(x[1], 0)"

# Define boundary condition
#bc_top = DirichletBC(V, Constant(25), top)
bc_bot = DirichletBC(V, Constant(10), bot)

bc = [bc_bot]

#Define variational problem
u = TrialFunction(V)
u_n = Function(V) # Define initial value to 0 by default
v = TestFunction(V)
f = Constant(0)
k = Constant(0.3)
g = Constant(-0.1)
r = Constant(-1)
s = Constant(30)

# Add Neumann and Robins condition in F function
F = u*v*dx + dt*k*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx + \
    g*v*ds(0) + r*(u-s)*v*ds(1)

a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile = File("result/solution.pvd")

# Time-stepping
u = Function(V)
t = 0.0

for n in range(num_steps):

    # Compute solution
    solve(a == L, u, bc)

    # Save to file and plot solution
    vtkfile << (u, t)

    # Update previous solution
    u_n.assign(u)

    # Update current time
    t += dt
