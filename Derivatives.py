import numpy as np 
from scipy import signal

class findDerivs:
    def __init__(self, I_gray):
        self.I_gray = I_gray
    
    def find_derivatives(self):
        """
        Take the input grey scale image, then find the change in the image
        on the x direction and the y direction.
        """
        # sobel operators
        sx = np.matrix("-1 0 1; -2 0 2; -1 0 1")
        sy = np.matrix("1 2 1; 0 0 0; -1 -2 -1")
        G = np.divide(
            np.matrix("2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2"), 159
        )

        Gx = signal.convolve2d(G, sx, "same")
        Gy = signal.convolve2d(G, sy, "same")

        Magx = signal.convolve2d(self.I_gray, Gx, "same")
        Magy = signal.convolve2d(self.I_gray, Gy, "same")

        Mag = np.sqrt((Magx * Magx) + (Magy * Magy))

        return Mag, Magx, Magy
    
    def calculate_orientation(self):
        Mag, Magx, Magy = self.find_derivatives()
        Ori = np.arctan2(Magy, Magx)
        return Ori