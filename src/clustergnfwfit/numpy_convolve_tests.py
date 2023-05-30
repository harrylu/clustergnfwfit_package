import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    arr = np.arange(36).reshape(6, 6)
    kernel = np.ones((2, 2))
    ret = scipy.signal.fftconvolve(arr, kernel, 'valid')
    plt.imshow(ret)
    plt.show()