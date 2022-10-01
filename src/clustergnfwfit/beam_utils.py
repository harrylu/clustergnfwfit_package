"""
Contains BeamHandler2D Class.
"""

import numpy as np
import scipy.fft
import scipy.interpolate
import astropy.convolution

class BeamHandler2D:
    """
    Class for working with the beams in the act auxilliary resources
    """

    def __init__(self, beam_path, beam_width):
        """Construct a BeamHandler2D.

        Args:
            beam_path (str): path of beam file
            beam_width (odd int): width of beam map (diameter)
        """

        self.BEAM_PATH = beam_path
        self.BEAM_WIDTH = beam_width
        # B(r) spline tck
        self.BEAM_SPLINE_TCK = self._read_beam_fourier(beam_path)
        self.BEAM_MAP = self._gen_beam_map(beam_width, self.BEAM_SPLINE_TCK)

    @staticmethod
    def _read_beam_fourier(beam_path):
        """Read in fourier frequency terms from beam files and represent as B-spline.

        Args:
            beam_path (str): path to the beam file

        Returns:
            tuple: A tuple (t,c,k) containing the vector of knots,
            the B-spline coefficients, and the degree of the spline.
            Can be evaluated using sp.interpolate.splrev.
        """
        # read in beam fourier and convert to spline; return B(r) spline tck
        beam_l, beam_Bl = [], []
        with open(beam_path, encoding="utf8") as f:
            beam_data = f.readlines()
            for line in beam_data:
                l, Bl = line.split()
                beam_l.append(float(l))
                beam_Bl.append(float(Bl))
        Br = scipy.fft.irfft(beam_Bl)
        Br_spline_tck = scipy.interpolate.splrep([i * 21.6 for i in range(len(Br))], Br)  # x points are in steps of 21.6"
        return Br_spline_tck

    @staticmethod
    def _gen_beam_map(width, beam_spline_tck):
        """Creates a 2d map of the beam by evaluating its B-spline representation
        at each pixel of the map.

        Args:
            width (odd int): width (diameter) of the map
            beam_spline_tck (tuple): A tuple (t,c,k) containing the vector of knots,
            the B-spline coefficients, and the degree of the spline.

        Raises:
            ValueError: Raises error if width is even. There needs to be a center pixel.

        Returns:
            2d array: Map of the beam.
        """
        if width % 2 == 0:
            raise ValueError('Argument width should be odd so that there is a center pixel.')

        # beam is azimuthal symmetric so only need to evaluate one quadrant
        # grid starts with row 0 col 0 at top left
        grid = np.empty((width // 2 + 1, width // 2 + 1))
        # we will make bottom right quadrant (actually only need part of the quadrant)
        # split quadrant into L shapes, evaluate vertical part of the L, fill in the rest of the L with the vertical values rotated cw 90 degrees
        for col_idx in range(grid.shape[1]):
            # calculate distance r in arcseconds for each pixel in the column (vertical part of the L shape)
            r = [np.linalg.norm([col_idx * 30, row_idx * 30]) for row_idx in range(col_idx, grid.shape[0])]
            Br = scipy.interpolate.splev(r, beam_spline_tck)
            # print('portion', grid[col_idx:, col_idx])
            grid[col_idx:, col_idx] = Br

            # fill in horizontal part of the L
            # print('portion2', grid[col_idx, col_idx:])
            grid[col_idx, col_idx:] = Br

        # plt.figure(0)
        # plt.title('Bottom right quadrant beam')
        # plt.imshow(grid, extent = (0, grid.shape[1], 0, grid.shape[0]) )

        # reflect left horizontally to get bottom left quadrant
        grid = np.pad(grid, [(0, 0), (grid.shape[0] - 1, 0)], 'reflect')
        # plt.figure(1)
        # plt.title('Bottom left quadrant beam')
        # plt.imshow(grid, extent = (0, grid.shape[1], 0, grid.shape[0]) )

        # reflect up vertically to get top two quadrants
        grid = np.pad(grid, [(grid.shape[0] - 1, 0), (0, 0)], 'reflect')
        # plt.figure(2)
        # plt.title('All quadrants beam')
        # plt.imshow(grid, extent = (0, grid.shape[1], 0, grid.shape[0]) )

        # plt.show()
        return grid

    def convolve2d(self, arr, cut_padding=False):
        """Does convolution with self's beam grid as kernel over the input 2d array with normalization.

        Args:
            arr (2d array): Array over which to do convolution
            cut_padding (bool, optional): Whether to cut off a number of pixels equal to (get_pad_pixels/2) from each side. Defaults to False.

        Returns:
            2d array: Convolved input array.
        """
        
        convolved = astropy.convolution.convolve(arr, self.BEAM_MAP, normalize_kernel=True)
        if cut_padding:
            half_pad = self.get_pad_pixels()//2
            convolved = convolved[half_pad:-half_pad, half_pad:-half_pad]
        return convolved

    def get_pad_pixels(self):
        """

        Returns:
            int: Number of pixels that should be applied as padding to input array pre-convolution.
        
        Notes:
            This will return 1 less than the value of the 2d beam map's width (diameter).
            The pre-convolved map should have (get_pad_pixels/2) pixels added to each side.

        """

        return self.BEAM_WIDTH - 1
