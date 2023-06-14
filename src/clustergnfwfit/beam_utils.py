"""
Contains BeamHandler Class.
"""

import numpy as np
import scipy.fft
import scipy.interpolate
import astropy.convolution
from astropy.modeling.functional_models import Gaussian2D

class BeamHandler:
    """
    Class for working with the beams in the act auxilliary resources
    Warning: Convolution is done with the beams at 30 arcsecond resolution for ACTPlanck and 20 arcseconds for Bolocam,
    so input should be 30 and 20 arcseconds, respectively.
    """

    def __init__(self, beam_map):
        """Construct a BeamHandler.

        Args:
            beam_map_width (odd int): width of beam map (diameter in pixels)
            beam_spline_tck (tuple (t,c,k)): A tuple (t,c,k) containing the vector of knots,
            the B-spline coefficients, and the degree of the spline that represents the radial function of the beam.
            Can be evaluated using sp.interpolate.splrev.
        """

        self.beam_map_width = beam_map.shape[0]
        if self.beam_map_width % 2 == 0:
            raise Exception("Beam map shape must be odd to have a center pixel")
        # B(r) spline tck
        self.beam_map = beam_map
    
    @staticmethod
    def rep_beam_spline(Bl):
        """Represents the beam as a spline. Lmax is assumed to be len(Bl) - 1

        Args:
            Bl (list of float): [Bl(l=0), Bl(l=1),.... Bl(l=lmax)]

        Returns:
            tuple: A tuple (t,c,k) containing the vector of knots,
            the B-spline coefficients, and the degree of the spline.
            Can be evaluated using sp.interpolate.splrev.
        """
        lmax = len(Bl) - 1
        Br = scipy.fft.irfft(Bl)
        delta_pix = 3600 * 180 / lmax
        Br_spline_tck = scipy.interpolate.splrep([i * delta_pix for i in range(len(Br))], Br)  # x points are in steps of delta_pix
        return Br_spline_tck

    @staticmethod
    def read_actplanck_beam_file(fpath):
        """Expects the file at fpath to be txt file containing l {space} Bl rows separated by newlines

        Args:
            fpath (string): path to beam file of specified format

        Returns:
            l, Bl
            both are lists
        """
        beam_l, beam_Bl = [], []
        with open(fpath, encoding="utf8") as f:
            beam_data = f.readlines()
            for line in beam_data:
                l, Bl = line.split()
                beam_l.append(float(l))
                beam_Bl.append(float(Bl))
        return beam_l, beam_Bl

    @staticmethod
    def gen_beam_map(width, beam_spline_tck):
        """Creates a 2d map of the beam by evaluating its B-spline representation
        at each pixel of the map. Each pixel is 30 arcseconds across.

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

    def convolve2d(self, arr, cut_padding=True):
        """Does convolution with self's beam grid as kernel over the input 2d array with normalization.

        Args:
            arr (2d array): Array over which to do convolution
            cut_padding (bool, optional): Whether to cut off a number of pixels equal to (get_pad_pixels/2) from each side. Defaults to False.

        Returns:
            2d array: Convolved input array.
        """
        
        convolved = astropy.convolution.convolve_fft(arr, self.beam_map, normalize_kernel=True)
        if cut_padding:
            half_pad = self.get_pad_pixels()//2
            convolved = convolved[half_pad:-half_pad, half_pad:-half_pad]
        return convolved

    def get_pad_pixels(self):
        """

        Returns:
            int: Number of extra (in addition to desired final data shape) pixels on each dimension
            that should be read in prior to convolution.
        
        Notes:
            This will return 1 less than the value of the 2d beam map's width (diameter).
            The pre-convolution image should read (get_pad_pixels/2) extra pixels on each side (4 sides).
            When the image is convolved by calling convolve2d with cut_padding = True,
            the extra pixels are discarded.

        """

        return self.beam_map_width - 1


class BeamHandlerACTPol(BeamHandler):
    def __init__(self, fpath, beam_map_width):
        """BeamHandler for Planck CMB found here:
        https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/previews/COM_CMB_IQU-commander_2048_R3.00_full/index.html

        Args:
            fpath (string): Expects the file at fpath to be txt file containing l {space} Bl rows separated by newlines
            beam_map_width (odd int): width of beam map (diameter in pixels)
        """
        _, beam_Bl = BeamHandler.read_actplanck_beam_file(fpath)
        self.beam_spline_tck = BeamHandler.rep_beam_spline(beam_Bl)
        beam_map = BeamHandler.gen_beam_map(beam_map_width, self.beam_spline_tck)
        super().__init__(beam_map)


class BeamHandlerPlanckCMB(BeamHandler):
    def __init__(self, Bl, beam_map_width):
        """BeamHandler for Planck CMB found here:
        https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/previews/COM_CMB_IQU-commander_2048_R3.00_full/index.html

        Args:
            Bl (_type_): list of [Bl(l=0), Bl(l=1), ..., Bl(l=lmax)]
            beam_map_width (odd int): width of beam map (diameter in pixels)
        """
        self.beam_spline_tck = BeamHandler.rep_beam_spline(Bl)
        beam_map = BeamHandler.gen_beam_map(beam_map_width, self.beam_spline_tck)
        super().__init__(beam_map)

class BeamHandlerBolocam(BeamHandler):
    def __init__(self, fwhm, beam_map_width):
        """BeamHandler for Bolocam (Gaussian beam)

        Args:
            fwhm (float): Full Width of Half Maximum for Gaussian beam in degrees.
            beam_map_width (int): Map width in pixels (must be odd)
        """
        bolocam_beam_sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        y, x = np.mgrid[:beam_map_width, :beam_map_width]
        x_mean = beam_map_width // 2
        y_mean = beam_map_width // 2
        bolocam_beam_sigma_pixels = bolocam_beam_sigma * 3600 / 20
        beam_map = Gaussian2D.evaluate(x, y, 1, x_mean, y_mean, bolocam_beam_sigma_pixels, bolocam_beam_sigma_pixels, 0)
        super().__init__(beam_map)