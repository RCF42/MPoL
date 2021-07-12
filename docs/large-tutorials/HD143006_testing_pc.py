# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup

# + cell_id="00001-be94721b-eee2-4e2e-96e2-dc65b9fd4f5b" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=663 execution_start=1623447169390 source_hash="4f0f20f8" tags=[]
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import download_file
from torch.utils import tensorboard

# downloading fits file
fname_F = download_file(
    "https://almascience.nrao.edu/almadata/lp/DSHARP/images/HD143006_continuum.fits",
    cache=True,
    pkgname="mpol",
)
# downloading extracted visibilities file
fname_EV = download_file(
    "https://zenodo.org/record/4904794/files/HD143006_continuum.npz",
    cache=True,
    pkgname="mpol",
)
# load extracted visibilities from npz file
dnpz = np.load(fname_EV)
uu = dnpz["uu"]
vv = dnpz["vv"]
weight = dnpz["weight"]
data = dnpz["data"]
# opening the fits file
dfits = fits.open(fname_F)
cdelt_scaling = dfits[0].header["CDELT1"] * 3600  # scaling [arcsec]
cell_size = abs(cdelt_scaling)  # [arcsec]
# close fits file
dfits.close()

# + cell_id="00008-d56e3afe-45cf-4fe5-a4e7-1803f28deec4" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=27 execution_start=1623441758279 source_hash="b76fed2d"
from mpol import gridding, coordinates

# creating Gridder object
coords = coordinates.GridCoords(cell_size=cell_size, npix=512)
gridder = gridding.Gridder(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data.real,  # separating the real and imaginary values of our data
    data_im=data.imag,
)

from mpol import (
    losses,  # here MPoL loss functions will be used
    connectors,  # required to calculate the residuals in log_figure
)

# -

img, beam = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/arcsec^2")

from mpol.precomposed import SimpleNet

model = SimpleNet(coords=coords, nchan=gridder.nchan)

import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# taking the dirty image and making it a tensor
dirty_image = torch.tensor(img.copy())

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # creating the optimizer
loss_fn = torch.nn.MSELoss()  # creating the MSEloss function from Pytorch

kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
snp = ax.imshow(np.squeeze(img), **kw)
ax.set_title("image")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(snp)


# +


def log_figure(
    model, residuals
):  # this function takes a snapshot of the image state, imaged residuals, amplitude of model visibilities, and amplitude of residuals

    # populate residual connector
    residuals()

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    im = ax[0, 0].imshow(
        np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=model.icube.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].imshow(
        np.squeeze(residuals.sky_cube.detach().cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=residuals.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow(
        np.squeeze(torch.log(model.fcube.ground_amp.detach()).cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(
        np.squeeze(torch.log(residuals.ground_amp.detach()).cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 1])

    return fig


# -


def train(model, dataset, optimizer, config, logevery=50):
    loss_tracker = []
    model.train()
    residuals = connectors.GriddedResidualConnector(model.fcube, dataset)
    for iteration in range(config["epochs"]):
        optimizer.zero_grad()
        vis = model.forward()  # get the predicted model
        sky_cube = model.icube.sky_cube
        # computing loss through MPoL loss function with more parameters
        loss = (
            losses.nll_gridded(vis, dataset)
            + config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
            + config["entropy"] * losses.entropy(sky_cube, config["prior_intensity"])
            + loss_fn(sky_cube, dirty_image)
        )
        loss_tracker.append(loss.item())

        if iteration % logevery == 0:
            writer.add_scalar("loss", loss.item(), iteration)
            writer.add_figure("image", log_figure(model, residuals), iteration)

        loss.backward()  # calculate gradient of the parameters
        optimizer.step()  # update the model parameters

    return loss.item()


dataset = (
    gridder.to_pytorch_dataset()
)  # export the visibilities from gridder to a PyTorch dataset


def test(model, dataset):
    model.eval()
    vis = model.forward()
    loss = losses.nll_gridded(
        vis, dataset
    )  # calculates the loss function that goes to make up the cross validation score
    return loss.item()


def cross_validate(model, config, k_fold_datasets):
    test_scores = []

    for k_fold, (train_dset, test_dset) in enumerate(k_fold_datasets):

        # create a new optimizer for this k_fold
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # train for a while
        train(model, train_dset, optimizer, config)
        # evaluate the test metric
        test_scores.append(test(model, test_dset))

    # aggregate all test scores and sum to evaluate cross val metric
    test_score = np.sum(np.array(test_scores))

    return test_score


from mpol import datasets

# +
# create a radial and azimuthal partition for the dataset
dartboard = datasets.Dartboard(coords=coords)

# create cross validator using this "dartboard"
k = 5
cv = datasets.KFoldCrossValidatorGridded(dataset, k, dartboard=dartboard, npseed=42)

# ``cv`` is a Python iterator, it will return a ``(train, test)`` pair of ``GriddedDataset``s for each iteration.
# Because we'll want to revisit the individual datasets
# several times in this tutorial, we're storing them into a list

k_fold_datasets = [(train, test) for (train, test) in cv]


# +
# %%time

# new directory to write the progress of our third cross val. loop to

new_config = (
    {  # config includes the hyperparameters used in the function and in the optimizer
        "lr": 0.3,
        "lambda_sparsity": 1.0e-3,
        "lambda_TV": 1.2e-4,
        "entropy": 1e-02,
        "prior_intensity": 2.0e-09,
        "epochs": 400,
    }
)

cvscore = cross_validate(model, new_config, k_fold_datasets)
print(f"Cross Validation Score: {cvscore}")

# +
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

imin, imax = np.amin(img), np.amax(img)

im = ax[0].imshow(
    np.squeeze(dirty_image.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
    vmin=imin,
    vmax=imax,
)

im = ax[1].imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
    vmin=imin,
    vmax=imax,
)

ax[0].set_xlim(left=0.75, right=-0.75)
ax[0].set_ylim(bottom=-0.75, top=0.75)
ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[0].set_title("Dirty Image")
ax[1].set_xlim(left=0.75, right=-0.75)
ax[1].set_ylim(bottom=-0.75, top=0.75)
ax[1].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[1].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[1].set_title("Image Cube")
plt.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.84, 0.17, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.tight_layout()
# -

"""
Total Squared Variation:

def TVSquare_image(sky_cube, epsilon=1e-10):

    .. math::

        L = \sum_{l,m,v} \sqrt{(I_{l + 1, m, v} - I_{l,m,v})^2 + (I_{l, m+1, v} - I_{l, m, v})^2 + \epsilon}


    # diff the cube in ll and remove the last row
    diff_ll = sky_cube[:, 0:-1, 1:] - sky_cube[:, 0:-1, 0:-1]

    # diff the cube in mm and remove the last column
    diff_mm = sky_cube[:, 1:, 0:-1] - sky_cube[:, 0:-1, 0:-1]

    loss = torch.sum((diff_ll ** 2 + diff_mm ** 2 + epsilon))

    return loss


"""
