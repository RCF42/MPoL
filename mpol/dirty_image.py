import matplotlib.pylab as plt
import numpy as np
import numpy.linalg as linalg
from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift, rfftfreq
from mpol.constants import *


def get_dirty_image(uu, vv, weights, re, im, cell_size, npix, **kwargs):
    r"""
    "Grid" the data visibilities using natural weighting, then do the inverse FFT. This delivers a maximum likelihood "dirty image."

    Args:
        uu (list): the uu points (in [:math:`k\lambda`])
        vv (list): the vv points (in [:math:`k\lambda`])
        weights (list): the thermal weights (in [:math:`\mathrm{Jy}^{-2}`])
        re (list): the real component of the visibilities (in [:math:`\mathrm{Jy}`])
        im (list): the imaginary component of the visibilities (in [:math:`\mathrm{Jy}`])
        cell_size (float): the image cell size (in arcsec)
        npix (int): the number of pixels in each dimension of the square image
        
    Returns:
        An image cube
        
    An image `cell_size` and `npix` correspond to particular `u_grid` and `v_grid` values from the RFFT. 

    """

    assert npix % 2 == 0, "Image must have an even number of pixels"

    # calculate the grid spacings
    cell_size = cell_size * arcsec  # [radians]
    # cell_size is also the differential change in sky angles
    # dll = dmm = cell_size #[radians]

    # the output spatial frequencies of the FFT routine
    uu_grid = np.fft.fftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]
    vv_grid = np.fft.fftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]

    nu = len(uu_grid)
    nv = len(vv_grid)

    du = np.abs(uu_grid[1] - uu_grid[0])
    dv = np.abs(vv_grid[1] - vv_grid[0])

    # expand and overwrite the vectors to include complex conjugates
    uu = np.concatenate([uu, -uu])
    vv = np.concatenate([vv, -vv])
    weights = np.concatenate([weights, weights])
    re = np.concatenate([re, re])
    im = np.concatenate([im, -im])  # the complex conjugates

    # The RFFT outputs u in the range [0, +] and v in the range [-, +],
    # but the dataset contains measurements at u [-,+] and v [-, +].
    # Find all the u < 0 points and convert them via complex conj
    # ind_u_neg = uu < 0.0
    # uu[ind_u_neg] *= -1.0  # swap axes so all u > 0
    # vv[ind_u_neg] *= -1.0  # swap axes
    # im[ind_u_neg] *= -1.0  # complex conjugate

    # calculate the sum of the weights within each cell
    # create the cells as edges around the existing points
    # note that at this stage, the bins are strictly increasing
    # when in fact, later on, we'll need to put this into fftshift format for the RFFT
    weight_cell, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=weights,
    )

    # calculate the weighted average and weighted variance for each cell
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    # also Bevington, "Data Reduction in the Physical Sciences", pg 57, 3rd ed.
    # where weight = 1/sigma^2

    # blank out the cells that have zero counts
    weight_cell[(weight_cell == 0.0)] = np.nan
    ind_ok = ~np.isnan(weight_cell)

    # weighted_mean = np.sum(x_i * weight_i) / weight_cell

    # first calculate np.sum(x_i * weight_i)
    real_part, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=re * weights,
    )

    imag_part, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=im * weights,
    )

    # divide by normalization weight_cell
    weighted_mean_real = real_part / weight_cell
    weighted_mean_imag = imag_part / weight_cell

    # do an fftshift on weighted_means and sigmas to get this into a 2D grid
    # that matches the RFFT output directly

    ind = np.fft.fftshift(ind_ok)  # RFFT indices that are not nan

    # gridded visibilities
    avg_re = np.fft.fftshift(weighted_mean_real)
    avg_im = np.fft.fftshift(weighted_mean_imag)

    # set the nans to zero
    avg_re[~ind] = 0.0
    avg_im[~ind] = 0.0

    # do the inverse FFT
    VV = avg_re + avg_im * 1.0j

    im = np.fliplr(np.fft.fftshift(np.fft.ifftn(VV, axes=(0, 1))))

    return np.real(im)
