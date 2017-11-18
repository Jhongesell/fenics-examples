from fenics import *
import time

T = 1.0
# final time
num_steps = 500
# number of time steps
dt = T / num_steps # time step size

# Create mesh
nx = ny = 50
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

#define function space
V = FunctionSpace(mesh, "P", 2)

# Define boundaries
# denics define natural Neumann boundaries by default
# so we dont need to specify right and left boundaries 
top   = "near(x[1], 1)"
bot   = "near(x[1], 0)"

# Define boundary condition
bc_top = DirichletBC(V, Constant(25), top)
bc_bot = DirichletBC(V, Constant(100), bot)

bc = [bc_top, bc_bot]

#Define variational problem
u = TrialFunction(V)
u_n = Function(V) # Define initial value to 0 by default
v = TestFunction(V)
f = Constant(0)
k = Constant(0.3)

F = u*v*dx + dt*k*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
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
