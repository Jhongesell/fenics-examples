from fenics import *
import time

T = 0.1
# final time
num_steps = 100
# number of time steps
dt = T / num_steps # time step size

# Create mesh
nx = ny = 100
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

#define function space
V = FunctionSpace(mesh, "P", 2)

# Define boundaries

top   = "near(x[1], 1)"
bot   = "near(x[1], 0)"
left  = "near(x[0], 1)"
right = "near(x[0], 0)"

# Define boundary condition
bc_top = DirichletBC(V, Constant(0), top)
bc_bot = DirichletBC(V, Constant(0), bot)
bc_left = DirichletBC(V, Constant(0), left)
bc_right = DirichletBC(V, Constant(0), right)

bc = [bc_top, bc_bot, bc_left, bc_right]

# Define initial value
u_0 = Expression("2.0*exp(-a*pow(x[0]-0.5, 2) - a*pow(x[1]-0.5, 2))", degree=2, a=10)
u_n = interpolate(u_0, V)

#Define variational problem
u = TrialFunction(V)
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
