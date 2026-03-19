import numpy as np

def generate_mesh(n):
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)

    nodes = np.array([[i, j] for j in y for i in x])

    elements = []
    for j in range(n):
        for i in range(n):
            n0 = j*(n+1) + i
            n1 = n0 + 1
            n2 = n0 + (n+1)
            n3 = n2 + 1

            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])

    return nodes, np.array(elements)


# 🔹 create mesh here
nodes, elements = generate_mesh(20)

def triangle_area(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    return 0.5 * abs(
        x1*(y2 - y3) +
        x2*(y3 - y1) +
        x3*(y1 - y2)
    )


def shape_function_gradients(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    b = np.array([
        y2 - y3,
        y3 - y1,
        y1 - y2
    ])

    c = np.array([
        x3 - x2,
        x1 - x3,
        x2 - x1
    ])

    return b, c

def element_stiffness(coords):
    A = triangle_area(coords)
    b, c = shape_function_gradients(coords)

    Ke = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            Ke[i, j] = (b[i]*b[j] + c[i]*c[j]) / (4*A)

    return Ke


# Test stiffness matrix for first triangle
coords = nodes[elements[0]]
Ke = element_stiffness(coords)

print("Element stiffness matrix:")
print(Ke)

# -----------------------------
# Stage 2: Global assembly
# -----------------------------
num_nodes = nodes.shape[0]
K_global = np.zeros((num_nodes, num_nodes))
F_global = np.zeros(num_nodes)

# -----------------------------
# Assemble stiffness matrix
# -----------------------------
for elem in elements:
    coords = nodes[elem]
    Ke = element_stiffness(coords)

    for i_local, i_global in enumerate(elem):
        for j_local, j_global in enumerate(elem):
            K_global[i_global, j_global] += Ke[i_local, j_local]

# -----------------------------
# Assemble load vector
# -----------------------------
'''
def source_function(x, y):
    return 10 * np.exp(-50 * ((x - 0.5)**2 + (y - 0.5)**2))
'''
f = 1.0  # constant heat source

for elem in elements:
    coords = nodes[elem]
    A = triangle_area(coords)
    Fe = np.ones(3) * (f * A / 3.0)

    for i_local, i_global in enumerate(elem):
        F_global[i_global] += Fe[i_local]

'''
for elem in elements:
    coords = nodes[elem]
    A = triangle_area(coords)

    Fe = np.zeros(3)

    for i_local, i_global in enumerate(elem):
        x, y = nodes[i_global]
        Fe[i_local] = source_function(x, y) * A / 3.0

    for i_local, i_global in enumerate(elem):
        F_global[i_global] += Fe[i_local]
'''
# -----------------------------
# Apply Dirichlet boundary conditions
# T = 0 on x = 0 boundary
# -----------------------------
fixed_nodes = [0, 3]
fixed_value = 0.0

for node in fixed_nodes:
    K_global[node, :] = 0.0
    K_global[:, node] = 0.0
    K_global[node, node] = 1.0
    F_global[node] = fixed_value

# -----------------------------
# Solve the system
# -----------------------------
T = np.linalg.solve(K_global, F_global)

# -----------------------------
# Output results
# -----------------------------
print("Nodal temperatures:")
for i, temp in enumerate(T):
    print(f"Node {i}: T = {temp:.6f}")


# -----------------------------
# Stage 3: Post-processing
# -----------------------------

# Store per-element quantities
grad_T = []
heat_flux = []
energy_density = []

k = 1.0  # thermal conductivity

for elem in elements:
    coords = nodes[elem]
    A = triangle_area(coords)
    b, c = shape_function_gradients(coords)

    # Nodal temperatures for this element
    T_elem = T[elem]

    # Temperature gradient (constant per element)
    dTdx = np.dot(b, T_elem) / (2.0 * A)
    dTdy = np.dot(c, T_elem) / (2.0 * A)

    grad = np.array([dTdx, dTdy])
    grad_T.append(grad)

    # Heat flux: q = -k * grad(T)
    q = -k * grad
    heat_flux.append(q)

    # Element energy density
    E = 0.5 * A * np.dot(grad, grad)
    energy_density.append(E)

grad_T = np.array(grad_T)
heat_flux = np.array(heat_flux)
energy_density = np.array(energy_density)

# -----------------------------
# Output results
# -----------------------------
for i in range(len(elements)):
    print(f"Element {i}:")
    print(f"  grad(T) = {grad_T[i]}")
    print(f"  heat flux = {heat_flux[i]}")
    print(f"  energy density = {energy_density[i]:.6f}")

# -----------------------------
# Derived quantities
# -----------------------------
grad_mag = np.linalg.norm(grad_T, axis=1)
total_energy = np.sum(energy_density)

print("\nGlobal diagnostics:")
print(f"Total thermal energy = {total_energy:.6f}")


import matplotlib.pyplot as plt
import matplotlib.tri as tri

triangulation = tri.Triangulation(nodes[:,0], nodes[:,1], elements)

plt.figure()
plt.tricontourf(triangulation, T, levels=20)
plt.colorbar(label="Temperature")
plt.title("Temperature Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.figure()
plt.tripcolor(triangulation, facecolors=grad_mag, edgecolors='k')
plt.colorbar(label="|∇T|")
plt.title("Temperature Gradient Magnitude")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.figure()
plt.tripcolor(triangulation, facecolors=energy_density, edgecolors='k')
plt.colorbar(label="Element Energy Density")
plt.title("Element-wise Energy Density")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


def run_fem(nodes, elements):
    num_nodes = len(nodes)
    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    for elem in elements:
        coords = nodes[elem]
        Ke = element_stiffness(coords)
        for i_l, i_g in enumerate(elem):
            for j_l, j_g in enumerate(elem):
                K[i_g, j_g] += Ke[i_l, j_l]

    f = 1.0
    for elem in elements:
        coords = nodes[elem]
        A = triangle_area(coords)
        Fe = np.ones(3) * (f * A / 3.0)
        for i_l, i_g in enumerate(elem):
            F[i_g] += Fe[i_l]

    fixed_nodes = np.where(nodes[:,0] == 0.0)[0]
    for node in fixed_nodes:
        K[node,:] = 0.0
        K[:,node] = 0.0
        K[node,node] = 1.0
        F[node] = 0.0

    T = np.linalg.solve(K, F)
    return T


mesh_sizes = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]
avg_gradients = []

for n in mesh_sizes:
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    nodes_ref = np.array([[i, j] for j in y for i in x])

    elements_ref = []
    for j in range(n):
        for i in range(n):
            n0 = j*(n+1) + i
            n1 = n0 + 1
            n2 = n0 + (n+1)
            n3 = n2 + 1
            elements_ref.append([n0, n1, n3])
            elements_ref.append([n0, n3, n2])

    elements_ref = np.array(elements_ref)
    T_ref = run_fem(nodes_ref, elements_ref)

    grads = []
    for elem in elements_ref:
        coords = nodes_ref[elem]
        A = triangle_area(coords)
        b, c = shape_function_gradients(coords)
        T_elem = T_ref[elem]
        gx = np.dot(b, T_elem)/(2*A)
        gy = np.dot(c, T_elem)/(2*A)
        grads.append(np.sqrt(gx**2 + gy**2))

    avg_gradients.append(np.mean(grads))


plt.figure()
plt.plot(mesh_sizes, avg_gradients, marker='o')
plt.xlabel("Mesh Resolution")
plt.ylabel("Average |∇T|")
plt.title("Mesh Sensitivity Study")
plt.grid(True)
plt.show()
