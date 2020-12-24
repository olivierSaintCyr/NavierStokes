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
    

    time_derivative = external_forces_map - \
                      grad_pressure + \
                      diffusion
    
    return time_derivative

def u(i : int, j : int, shape : tuple):
    return -np.cos(2*np.pi/shape[0]*i)*np.cos(2*np.pi/shape[1]*j)

def gen_U(shape : tuple):
    return np.fromfunction(lambda i, j: u(i, j, shape=shape), shape, dtype=float) 

def v(i : int, j : int, shape : tuple):
    return -np.sin(2*np.pi/shape[0]*i)*np.sin(2*np.pi/shape[1]*j)

def gen_V(shape : tuple):
    return np.fromfunction(lambda i, j: v(i, j, shape=shape), shape, dtype=float)

def dynamic_pressure(U : np.array, V : np.array, density : float):
    return 1/2 * density * (np.square(U) + np.square(V))

def divergence(U : np.array, V : np.array, delta_x : float, delta_y : float):
    return derive_map(U, delta_space= delta_x , axis=0) + derive_map(V, delta_space= delta_y , axis=1)

if __name__ == "__main__":
    shape  = (100, 100)
    delta_x, delta_y = 2*np.pi/shape[0], 2*np.pi/shape[1]
    delta_t = 0.01
    viscosity = 0.0001
    density = 1.0

    shape = (100,100)

    P = np.zeros(shape)

    Gx = np.fromfunction(lambda i, j: np.sin(2*np.pi/shape[0]*i), shape, dtype=float)
    Gy = np.zeros(shape)

    U = gen_U(shape)
    V = gen_V(shape)

    fig, ax = plt.subplots()

    x,y = np.meshgrid(np.linspace(0,2*np.pi,shape[0]) ,np.linspace(0,2*np.pi,shape[1]))
    ax.streamplot(x,y,U,V)
    plt.show()

    for t in range(1000):
        if (t%10==0):
            print('\n')
            print(t * delta_t)
            print('\n')
            print(V)
            fig, ax = plt.subplots()
            ax.streamplot(x,y,U,V)
            plt.show()
        
        U = time_derivative_map(U, V, P, Gx, delta_x, delta_y, density, viscosity, axis = 1) * delta_t + U
        V = time_derivative_map(V, U, P, Gy, delta_y, delta_x, density, viscosity, axis = 0) * delta_t + V
