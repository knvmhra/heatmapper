import CoreWLAN
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Kernel
import matplotlib.pyplot as plt

from typing import Tuple

def get_rssi():
    cwlan_interface = CoreWLAN.CWInterface.interface()
    return cwlan_interface.rssiValue()

class WifiKernel(Kernel):
    def __init__(self, router_pos) -> None:
        assert(len(router_pos) == 2)
        self.router_pos = router_pos

    def diag(self, X):
        return np.ones(X.shape[0])
    
    def is_stationary(self):
        return False
    
    def get_params(self, deep=True):
        return {"router_pos": self.router_pos}

    def set_params(self, **params):
        if "router_pos" in params:
            self.router_pos = np.array(params["router_pos"])
        return self
    
    def clone_with_theta(self, theta):
        return self

    def __call__(self, X, Y=None, eval_gradient= False):
        if Y is None:
            Y = X
        
        router_pos = np.array(self.router_pos)

        X_router_dist = np.sqrt(np.sum((X - self.router_pos)**2, axis=1))
        Y_router_dist = np.sqrt(np.sum((Y - self.router_pos)**2, axis=1))

        # using path loss to scale signal strength with distance from router
        X_path_loss = -20 * np.log10(X_router_dist + 0.1)
        Y_path_loss = -20 * np.log10(Y_router_dist + 0.1)

        K = np.outer(X_path_loss, Y_path_loss)

        if eval_gradient:
            return K, np.empty((X.shape[0], Y.shape[0], 0))

        return K

def generate_synthetic_wifi_data(router_pos: Tuple[float, float], room_bounds: Tuple[float, float, float, float], n_points = 50):
    
    # Generate synthetic measurements
    np.random.seed(42)
    n_points = 30
    X = np.random.uniform(room_bounds[0], room_bounds[1], (n_points, 1))
    Y = np.random.uniform(room_bounds[2], room_bounds[3], (n_points, 1))
    points = np.hstack([X, Y])
    
    # Calculate synthetic RSSI values
    distances = np.sqrt(np.sum((points - router_pos)**2, axis=1))
    base_signal = -40  # dBm at 1m
    path_loss = -20 * np.log10(distances)  # Free space path loss
    rssi = base_signal + path_loss + np.random.normal(0, 2, n_points)
    
    # Create measurements list
    measurements = [(x, y, r) for (x,y),r in zip(points, rssi)]

    return measurements

def create_room_grid(bounds, grid_size):
    x_min, x_max, y_min, y_max = bounds

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)

    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    return grid_points, xx, yy

def get_length_scale(bounds):
    x_min, x_max, y_min, y_max = bounds
    x_size = x_max - x_min
    y_size = y_max - y_min
    
    return (x_size + y_size) / 10

def create_heatmap(measurements: Tuple[float, float, float], router_pos: Tuple[float, float], room_bounds: Tuple[float, float, float, float], grid_size: float):
    X = np.array([(x, y) for x, y , _ in measurements])
    rssi = np.array([r for _, _, r in measurements])

    grid_points, xx, yy = create_room_grid(room_bounds, grid_size)

    length_scale = get_length_scale
    wifi_kernel = WifiKernel(router_pos)
    noise_kernel = WhiteKernel()
    kernel = wifi_kernel + noise_kernel

    gpr = GaussianProcessRegressor(kernel= kernel)
    gpr.fit(X, rssi)

    r_hat = gpr.predict(grid_points).reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize = (10, 8))

    im = ax.imshow(r_hat, extent= list(room_bounds), origin = 'lower', cmap= 'RdYlBu_r', vmin= -90, vmax = -30)

    ax.scatter(X[:, 0], X[:, 1], c= 'black', s= 20, alpha= 0.5, label= 'Measurements')

    ax.scatter(router_pos[0], router_pos[1], c = 'red', s = 100, marker= "*", label = 'Router')

    plt.colorbar(im, label = 'Signal Strength (dBm)')
    ax.legend()

    return fig

def main():
    # Load measurements
    measurements = np.loadtxt('measurements.csv', delimiter=',', skiprows= 1)
    
    # Load router position
    router_pos = tuple(np.loadtxt('router.csv', delimiter=',', skiprows=1))
    # Get room bounds from actual data
    room_bounds = (
        np.min(measurements[:, 0]),  # x min
        np.max(measurements[:, 0]),  # x max
        np.min(measurements[:, 1]),  # y min
        np.max(measurements[:, 1])   # y max
    )
    
    # Convert measurements to list of tuples for your existing code
    measurements_list = [tuple(m) for m in measurements]
    
    # Create heatmap with floorplan background
    fig = create_heatmap(
        measurements=measurements_list,
        router_pos=router_pos,
        room_bounds=room_bounds,
        grid_size=50
    )
    
    # Add floorplan as background
    floorplan = plt.imread('B1.png')
    ax = fig.gca()
    ax.imshow(floorplan, extent=list(room_bounds), alpha=0.5)
    
    plt.show()

if __name__ == "__main__": 
    main()








