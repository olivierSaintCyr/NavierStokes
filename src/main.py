import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import NavierStokesEquation as NSE

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

        self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.x, self.y = np.meshgrid(np.linspace(0,2*np.pi,self.fluid.shape[0]) ,
                                     np.linspace(0,2*np.pi,self.fluid.shape[1]))
        
        self.quiver = self.ax[0].quiver(self.x, self.y, self.fluid.U, self.fluid.V)
        self.pressure_hm = self.ax[1].imshow(NSE.dynamic_pressure(self.fluid.U, self.fluid.V, self.fluid.density), cmap='magma')

    def update_plot(self, Q):
        self.fluid.update_UV()
        self.quiver.set_UVC(self.fluid.U, self.fluid.V)
        self.pressure_hm.set_data(NSE.dynamic_pressure(self.fluid.U, self.fluid.V, self.fluid.density))
        return self.quiver,
    
    def tight_layout(self):
        self.fig.tight_layout()
    
    
if __name__ == "__main__":
    shape = (200,200)
    delta_x, delta_y = 2*np.pi/shape[0], 2*np.pi/shape[1]
    delta_t = 0.01
    viscosity = 0.00005
    density = 1.0

    fluid = Fluid(shape, delta_x, delta_y, delta_t, density, viscosity)
    figure = Figure(fluid)

    anim = animation.FuncAnimation(figure.fig, figure.update_plot, fargs=(),
                                interval=50)

    figure.tight_layout()
    plt.show()
