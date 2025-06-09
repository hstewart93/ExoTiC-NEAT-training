#!/usr/bin/python
# -*- coding: latin-1 -*-
"""
A module to make a bunch of SOSS simulations with randomized order 0 contaminants

Example Usage:
    import soss_simulations as ss
    scene, sources = ss.simulate_soss()
    ss.run_simulations(100, '100_soss_sims.h5')

    # To randomize the target params do:
    ss.run_simulations(targ_Jmag=(0, 10), targ_Teff=(2000, 10000), N_contaminants=(0, 10))

Then to read the data of index 0 with the list of its contaminant [y, x, Jmag] values, do:
    import h5py
    with h5py.File("soss_simulations.h5", "r") as f:
        contaminants = f["meta_0"][:]
        simulation_0 = f["data_0"][:]

"""
from concurrent.futures import ProcessPoolExecutor
import random

from astropy.io import fits
import numpy as np
import h5py
from bokeh.plotting import figure, show
from bokeh.models import LogColorMapper, LinearColorMapper
from exoctk.contam_visibility import field_simulator as fs


def find_order0s(rate_file, aperture='NIS_SUBSTRIP256', plot=True):
    """
    Read in a FITS file and identify all the order 0 contaminants in the image

    Parameters
    ----------
    rate_file: str
        The rate file of the observation to load
    plot: bool
        Plot the data along with the identified order 0 contaminants
    """
    # Get the data
    data = fits.getdata(rate_file)
    V3PA = fits.getval(rate_file, 'PWCPOS')
    ra = fits.getval(rate_file, 'RA')
    dec = fits.getval(rate_file, 'DEC')

    # Get the field of sources
    sources = fs.find_sources(ra, dec)

    # Make the V3PA simulation for the given data
    if plot:
        result, plt = fs.calc_v3pa(V3PA, sources, aperture, data=data, plot=True)
        show(plt)
    else:
        result = fs.calc_v3pa(V3PA, sources, aperture, data=data, plot=False)

    return result, sources


def run_simulations(N_simulations=10, output_file='soss_simulations.h5', targ_Teff=6000, targ_Jmag=9, N_contaminants=5,
                    Jmag_range=(1, 16), aperture='NIS_SUBSTRIP256'):
    """
    Runs multiple simulations in parallel with varying N values and stores results in an HDF5 file.

    To read the data, do:
    with h5py.File("soss_simulations.h5", "r") as f:
        contaminants = f["meta_0"][:]
        simulation_0 = f["data_0"][:]
    """
    # Randomize number of contaminants
    if isinstance(N_contaminants, int):
        contaminants = [N_contaminants] * N_simulations
    elif isinstance(N_contaminants, (tuple, list)):
        contaminants = np.random.randint(N_contaminants[0], N_contaminants[1], N_simulations)
    elif isinstance(N_contaminants, np.ndarray):
        contaminants = N_contaminants
    else:
        raise ValueError("Please pass an integer for uniform, tuple for random sampling, or an array of specific N_contaminants.")

    # Randomize target Teff
    if isinstance(targ_Teff, (int, float)):
        teffs = [targ_Teff] * N_simulations
    elif isinstance(targ_Teff, (tuple, list)):
        teffs = np.random.randint(targ_Teff[0], targ_Teff[1], N_simulations)
    elif isinstance(targ_Teff, np.ndarray):
        teffs = targ_Teff
    else:
        raise ValueError("Please pass an integer for uniform, tuple for random sampling, or an array of specific targ_Teff.")

    # Randomize target Jmag
    if isinstance(targ_Jmag, (int, float)):
        Jmags = [targ_Jmag] * N_simulations
    elif isinstance(targ_Jmag, (tuple, list)):
        Jmags = np.round(np.random.uniform(targ_Jmag[0], targ_Jmag[1], N_simulations), 2)
    elif isinstance(targ_Jmag, np.ndarray):
        Jmags = targ_Jmag
    else:
        raise ValueError("Please pass an integer for uniform, tuple for random sampling, or an array of specific targ_Jmag.")

    print("Generating SOSS simulations with the following parameters:")
    print("#    Targ_Teff Targ_Jmag N_contaminants")
    for i, (n, t, j) in enumerate(zip(contaminants, teffs, Jmags)):
        print(f"{i:<4} {t:<9} {j:<11} {n:<14}")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            simulate_soss,
            teffs,
            Jmags,
            contaminants,
            [Jmag_range] * N_simulations,
            [aperture] * N_simulations
        ))

    simulation_data = np.array([r[0] for r in results])
    meta_contaminants = np.array([r[1] for r in results], dtype="object")
    simulation_clean = np.array([r[2] for r in results])
    simulation_contaminant = np.array([r[3] for r in results])
    meta_target = np.array([teffs, Jmags]).T

    with h5py.File(output_file, "w") as f:
        f.create_dataset("simulation_data", data=simulation_data, compression="gzip")
        f.create_dataset("simulation_clean", data=simulation_clean, compression="gzip")
        f.create_dataset("simulation_contaminant", data=simulation_contaminant, compression="gzip")

    np.save(output_file.replace(".h5", "_meta_contaminants.npy"), meta_contaminants)
    np.save(output_file.replace(".h5", "_meta_target.npy"), meta_target)


    print("Results saved to", output_file)


def simulate_soss(targ_Teff=6000, targ_Jmag=9, N_contaminants=5, Jmag_range=(1, 16), aperture='NIS_SUBSTRIP256',
                  scale='linear', norm=1000, plot=False):
    """Produce a contamination field simulation at the given sky coordinates

    Parameters
    ----------
    targ_Teff: int
        The Teff of the target
    targ_Jmag: float
        The Jmag of the target
    n_sims: int
        The number of simulations
    contaminants_range: tuple
        The range of the number of contaminants to sample from
    Teff_range: tuple
        The range of Teff values to sample from
    Jmag_range: tuple
        The range of Jmag values to sample from
    aperture: str
        The name of the aperture to use, ['NIS_SUBSTRIP256', 'NIS_SUBSTRIP96', 'NIS_FULL']

    Returns
    -------
    simuCube : np.ndarray
        The simulated data cube. Index 0 and 1 (axis=0) show the trace of
        the target for orders 1 and 2 (respectively). Index 2-362 show the trace
        of the target at every position angle (PA) of the instrument.
    plt: NoneType, bokeh.plotting.figure
        The plot of the contaminationas a function of PA

    Example
    -------
    from exoctk.contam_visibility import field_simulator as fs
    ra, dec = 91.872242, -25.594934
    targframe, starcube, results = fs.field_simulation(ra, dec, 'NIS_SUBSTRIP256')
    """
    # Make a blank scene
    scene = np.zeros((96 if aperture == 'NIS_SUBSTRIP96' else 256, 2048)).astype(np.float32)
    contaminant_scene = scene.copy()

    # Get the order 1/2/3 traces and scale
    trace_o1, trace_o2, trace_o3 = fs.get_trace(aperture, targ_Teff, 'STAR')

    # Orient traces
    trace_o1 = np.rot90(trace_o1.T[:, ::-1] * 1.5, k=1)  # Scaling factor based on observations
    trace_o2 = np.rot90(trace_o2.T[:, ::-1] * 1.5, k=1)  # Scaling factor based on observations
    trace_o3 = np.rot90(trace_o3.T[:, ::-1] * 1.5, k=1)  # Scaling factor based on observations

    # Pad or trim SUBSTRIP256 simulation for SUBSTRIP96 or FULL frame
    if aperture == 'NIS_SOSSFULL':
        trace_o1 = np.pad(trace_o1, ((1792, 0), (0, 0)), 'constant')
        trace_o2 = np.pad(trace_o2, ((1792, 0), (0, 0)), 'constant')
        trace_o3 = np.pad(trace_o3, ((1792, 0), (0, 0)), 'constant')
    elif aperture == 'NIS_SUBSTRIP96':
        trace_o1 = trace_o1[:96, :]
        trace_o2 = trace_o2[:96, :]
        trace_o3 = trace_o3[:96, :]

    # Scale and place the order 1/2/3 target traces in the scene
    scene += trace_o1 * targ_Jmag * norm
    scene += trace_o2 * targ_Jmag * norm
    scene += trace_o3 * targ_Jmag * norm

    clean_scene = scene.copy()
    # Get the order 0 stamp
    order0 = fs.get_order0(aperture) * 1e7 # Scaling factor based on observations

    # Add the random order 0s
    target_rows, target_cols = scene.shape
    array_rows, array_cols = order0.shape
    contam_list = []
    for _ in range(N_contaminants):

        # Randomize the top-left position of the array of ones
        start_row = random.randint(-array_rows + 1, target_rows)
        start_col = random.randint(700, target_cols) # POM starts at col 700

        # Randomly choose a multiplication factor from the specified range
        factor = random.uniform(*Jmag_range)

        # Determine overlap region in the target array
        end_row = start_row + array_rows
        end_col = start_col + array_cols

        # Clip indices to be within the bounds of the target array
        target_start_row = max(start_row, 0)
        target_start_col = max(start_col, 0)
        target_end_row = min(end_row, target_rows)
        target_end_col = min(end_col, target_cols)

        # Determine the corresponding region of the array of ones
        array_start_row = max(0, -start_row)
        array_start_col = max(0, -start_col)
        array_end_row = array_start_row + (target_end_row - target_start_row)
        array_end_col = array_start_col + (target_end_col - target_start_col)

        # Save the info used for wach contaminant
        contam_list.append([start_col + 24, start_row + 24, factor])

        # Place the scaled array of ones in the target array
        scene[target_start_row:target_end_row, target_start_col:target_end_col] += \
            factor * order0[array_start_row:array_end_row, array_start_col:array_end_col]

        # Place the scaled array of ones in the empty target array
        contaminant_scene[target_start_row:target_end_row, target_start_col:target_end_col] += \
            factor * order0[array_start_row:array_end_row, array_start_col:array_end_col]

    if plot:
        # Make the plot
        plt = figure(title=f"{N_contaminants} Contaminants", width=900, height=300,
                      x_range=(0, scene.shape[1]), y_range=(0, scene.shape[0]),
                      tools="pan,wheel_zoom,reset", toolbar_location="above")

        # Use image to plot the array
        if scale == 'log':
            scene[scene <= 0] = 1e-6
            mapper = LogColorMapper(palette="Viridis256", low=scene.min(), high=scene.max())
        else:
            mapper = LinearColorMapper(palette="Viridis256", low=scene.min(), high=scene.max())

        plt.image(image=[scene], x=0, y=0, dw=scene.shape[1], dh=scene.shape[0], color_mapper=mapper)
        xcoords, ycoords, mags = np.array(contam_list).T
        plt.circle(xcoords, ycoords, size=20, fill_color=None, line_color='red')

        show(plt)

    return scene, contam_list, clean_scene, contaminant_scene
