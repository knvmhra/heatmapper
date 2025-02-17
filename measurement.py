import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from wifi_scanner import get_rssi
import os

class MeasurementCollector:
    def __init__(self, image_path: str):
        self.image = plt.imread(image_path)
        self.measurements: List[Tuple[float, float, float]] = []
        self.router_pos = None
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title('Click to set router position, then click points to measure')
        self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('key_press_event', 
                                  lambda event: plt.close() if event.key == 'q' else None)
    
    def _onclick(self, event):
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        
        if self.router_pos is None:
            self.router_pos = (x, y)
            self.ax.plot(x, y, 'r*', markersize=15, label='Router')
            self.ax.legend()
            self.ax.set_title('Click points to measure (press q when done)')
            self.fig.canvas.draw_idle()
            return
        
        rssi = get_rssi()
        self.measurements.append((x, y, rssi))
        self.ax.plot(x, y, 'bo', alpha=0.5)
        print(f'Added measurement: ({x:.1f}, {y:.1f}, {rssi})')
        self.fig.canvas.draw_idle()
    
    def save(self, measurements_path='measurements.csv', router_path='router.csv'):
        if not self.measurements:
            return
            
        # Save measurements (append mode)
        header = 'x,y,rssi' if not os.path.exists(measurements_path) else ''
        with open(measurements_path, 'a') as f:
            if header:
                f.write(header + '\n')
            np.savetxt(f, self.measurements, delimiter=',')
        
        # Save router position (new file each time since it's a single point)
        np.savetxt(router_path, [self.router_pos], 
                  delimiter=',', header='x,y', comments='')
        
        print(f'Saved {len(self.measurements)} measurements')

def main():
    collector = MeasurementCollector('B1.png')
    plt.show()
    collector.save()

if __name__ == '__main__':
    main()