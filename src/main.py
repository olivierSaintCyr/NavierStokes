import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import NavierStokesEquation as NSE

# shape  = (100, 100)
# U = NSE.gen_U(shape)
# V = NSE.gen_V(shape)

# def update_streamplot(S, pressure_map : np.array, 
#                       Gx : np.array, Gy : np.array,
#                       delta_x : float, delta_y : float , 
#                       delta_t : float, density : float, 
#                       viscosity : float):

#     U = NSE.time_derivative_map(U, V, P, Gx, delta_x, delta_y, density, viscosity, axis = 1) * delta_t + U
#     V = NSE.time_derivative_map(V, U, P, Gy, delta_y, delta_x, density, viscosity, axis = 0) * delta_t + V
#     S.set_UVC(U, V)
#     return S,

class Fluid:
    def __init__(self, shape, delta_x, delta_y, delta_t, density, viscosity):
        self.shape = shape
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_t = delta_t
        self.density = density
        self.viscosity = viscosity

        self.U = NSE.gen_U(shape)
        self.V = NSE.gen_V(shape)

        self.P = np.zeros(shape)
        self.Gx = np.fromfunction(lambda i, j: np.sin(2*np.pi/shape[0]*i), shape, dtype=float)
        self.Gy = np.zeros(shape)

    def update_UV(self):
        self.U = NSE.time_derivative_map(self.U, self.V, self.P, self.Gx, self.delta_x, self.delta_y, self.density, self.viscosity, axis = 1) * self.delta_t + self.U
        self.V = NSE.time_derivative_map(self.V, self.U, self.P, self.Gy, self.delta_y, self.delta_x, self.density, self.viscosity, axis = 0) * self.delta_t + self.V

class Figure:
    def __init__(self, fluid : Fluid):
        self.fluid = fluid

        self.fig, self.ax = plt.subplots()

        self.x, self.y = np.meshgrid(np.linspace(0,2*np.pi,self.fluid.shape[0]) ,
                                     np.linspace(0,2*np.pi,self.fluid.shape[1]))
        
        self.streamplot = self.ax.quiver(self.x, self.y, self.fluid.U, self.fluid.V)

    def update_streamplot(self, Q):
        self.fluid.update_UV()
        self.streamplot.set_UVC(self.fluid.U, self.fluid.V)
        return self.streamplot,
    
    def tight_layout(self):
        self.fig.tight_layout()
    
    
if __name__ == "__main__":
    shape = (100,100)
    delta_x, delta_y = 2*np.pi/shape[0], 2*np.pi/shape[1]
    delta_t = 0.01
    viscosity = 0.0001
    density = 1.0

    fluid = Fluid(shape, delta_x, delta_y, delta_t, density, viscosity)
    figure = Figure(fluid)

    anim = animation.FuncAnimation(figure.fig, figure.update_streamplot, fargs=(),
                                interval=50)

    figure.tight_layout()
    plt.show()



    # P = np.zeros(shape)

    # Gx = np.fromfunction(lambda i, j: np.sin(2*np.pi/shape[0]*i), shape, dtype=float)
    # Gy = np.zeros(shape)
    
    # anim = animation.FuncAnimation(fig, update_streamplot, fargs=(S, P, Gx, Gy, delta_x, ),
    #                            interval=50, blit=False)
    # fig.tight_layout()
    # plt.show()
    # fig, ax = plt.subplots()
    # x,y = np.meshgrid(np.linspace(0,2*np.pi,shape[0]) ,np.linspace(0,2*np.pi,shape[1]))
    # ax.streamplot(x,y,U,V)
    # plt.show()