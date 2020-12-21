import numpy as np
import matplotlib.pyplot as plt
# Definition a map is a scalar at a certain location for exemple
# a pressure map is a matrix which collum corresponds to the 
# x coordinates and the rows the y coordinnates.

# Need to add du/di in the equation

# Compute all discret derivative over a map of value in 2 dimension
def derive_map(value_map : np.array, delta_space : float, axis = None):
    dmap = value_map - np.roll(value_map, 1, axis = axis)
    dmap_dspace = dmap / delta_space
    return dmap_dspace

# Compute all discret second derivative over a map of value in 2 dimension
def second_derive_map(value_map : np.array, delta_space : float, axis = None):
    d2map = value_map - 2 * np.roll(value_map, 1, axis = axis) + \
            np.roll(value_map, 2, axis = axis)

    d2map_dspace2 = d2map / (delta_space*delta_space) 
    return d2map_dspace2

# Computation of the time derivative of an entire map using Navier Stokes equation
# over a discrete map
def time_derivative_map(velocity_map : np.array, perp_velocity_map : np.array, 
                        pressure_map : np.array, external_forces_map : np.array, 
                        delta_space : float, perp_delta_space : float , 
                        density : float, viscosity : float, axis = None):
    
    perp_axis = (axis + 1)%2

    grad_pressure = derive_map(pressure_map, delta_space, axis = axis) / density

    diffusion_const = viscosity / density
    diffusion = diffusion_const * \
                (second_derive_map(velocity_map, delta_space, axis=axis) +  \
                 second_derive_map(velocity_map, perp_delta_space, axis = perp_axis))
    
    variation_space = np.multiply(
                            derive_map(velocity_map, delta_space, axis = axis), 
                            velocity_map
                            )
    
    variation_perp_space = np.multiply(
                                derive_map(velocity_map, perp_delta_space, axis = perp_axis),
                                perp_velocity_map
                                )

    time_derivative = external_forces_map - \
                      grad_pressure + \
                      diffusion - \
                      variation_space - \
                      variation_perp_space
    
    return time_derivative

# Need to make space contious
def i_to_y(i : int):
    return np.sin(np.pi/100 * i)

def j_to_x(j : int):
    return np.sin(np.pi/100 * j)

# j = x, i = y
def gen_U(shape : tuple):
    return np.fromfunction(lambda i, j: j_to_x(j)*j_to_x(j), shape, dtype=float) 

def gen_V(shape : tuple):
    return np.fromfunction(lambda i, j: -j_to_x(j)*i_to_y(i), shape, dtype=float)


if __name__ == "__main__":
    delta_x, delta_y = 3, 3
    delta_t = 0.01
    viscosity = 100
    density = 1.0

    shape = (100,100)

    P = np.zeros(shape)

    Gx = np.zeros(shape)
    Gy = np.zeros(shape)

    # U = np.arange(9.0).reshape(3,3)

    # dU/dx = - dV/dy
    U = gen_U(shape)
    V = gen_V(shape)

    fig, ax = plt.subplots()
    #im = ax.imshow(derive_map(V, delta_y, axis = 0))
    im = ax.imshow(V)
    plt.show()

    print(U)
    print(V)
    for t in range(9000):
        if (t == 10 or t == 11 or t == 50):
            fig, ax = plt.subplots()
            im = ax.imshow(V)
            plt.show()
        print(t)
        print('\n')
        
        U = time_derivative_map(U, V, P, Gx, delta_x, delta_y, density, viscosity, axis = 1) * delta_t + U
        V = time_derivative_map(V, U, P, Gy, delta_y, delta_x, density, viscosity, axis = 0) * delta_t + V
        print(U)
        print(V)
        print('\n')
        print('\n')



    

     
                      
                      
    

