import numpy as np 
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import time
from tqdm import tqdm
import ast
import sys

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


def read_config (filename):
    """
    This function reads in the config file provided by the user
    """
    config = {}

    with open(filename, "r") as f:

        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # Ignore blank lines and comments
        if not line or line.startswith("#"):
            i += 1
            continue

        if "=" not in line:
            raise ValueError(f"Invalid config line: {line}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Handle multiline values, e.g. file_templates
        while value.count("[") > value.count("]"):
            i += 1
            value += lines[i].strip()

        try:
            config[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            config[key] = value

        i += 1

    return config

def read_data(path):
    """
    Read one reduction file.

    Inputs:
    path : path to the reduction file

    Output:
    MJD : array of modified Julian Dates
    phase : array of orbital phases from the reduction file
    mag : array of magnitudes
    err : array of magnitude errors
    """

    data = np.genfromtxt(path)

    MJD = data[:, 0]
    phase = data[:, 1]
    mag = data[:, 2]
    err = data[:, 3]

    return MJD, phase, mag, err

def fit_baseline_model(MJD, PB, tasc, mag, mag_err):
    """
    Fit the baseline model: A sin(2πφ) + B cos(4πφ) + C
    for fixed orbital period and tasc.

    Inputs:
    MJD : array of modified Julian Dates
    PB: orbital period
    tasc: epoch of ascending node
    mag : array of magnitudes
    err : array of magnitude errors

    Outputs:
    A : irradiation coefficient
    B : ellipsoidal coefficient
    C : constant offset
    chi2 : chi-squared of the fit
    model : model evaluated at the input phases.
    phase : array of orbital phases
    irradiation : irradiation component
    ellipsoidal : ellipsoidal component
    """

    # Compute orbital phase
    phase = ((MJD - tasc) / PB) % 1

    # Create design matrix
    X = np.column_stack([np.sin(2 * np.pi * phase), np.cos(4 * np.pi * phase), np.ones_like(MJD)])

    # Weighted least squares
    X_weight = X / mag_err[:, None]
    mag_weight = mag / mag_err

    beta, _, _, _ = np.linalg.lstsq(X_weight, mag_weight, rcond=None)

    A, B, C = beta
    model = X @ beta

    irradiation = A * np.sin(2 * np.pi * phase)
    ellipsoidal = B * np.cos(4 * np.pi * phase)

    chi2 = np.sum(((mag - model) / mag_err) ** 2)

    return (A, B, C, chi2, model, phase, irradiation, ellipsoidal)

def fit_band_baseline(paths, PB, tasc, band):
    """
    Find the baseline model for all observation runs in one band.

    Inputs:
    paths : paths to the reduction files for this band
    PB : orbital period
    tasc : epoch of ascending node
    band : band

    Outputs:
    A : best-fit irradiation coefficient for baseline model
    B : best-fit ellipsoidal coefficient for baseline model
    C : best-fit constant offset for baseline model
    """

    data_MJD = []
    data_phase = []
    data_mag = []
    data_err = []

    # Read every observation
    for path in paths:

        MJD, phase, mag, err = read_data(path)

        data_MJD.append(MJD)
        data_phase.append(phase)
        data_mag.append(mag)
        data_err.append(err)

    # Concatenate into single arrays
    data_MJD = np.concatenate(data_MJD)
    data_phase = np.concatenate(data_phase)
    data_mag = np.concatenate(data_mag)
    data_err = np.concatenate(data_err)

    # Sort by phase
    idx = np.argsort(data_phase)

    data_phase = data_phase[idx]
    data_mag = data_mag[idx]
    data_err = data_err[idx]
    data_MJD = data_MJD[idx]

    # Initial baseline fit
    A, B, C, chi2, model, phase, irradiation,ellipsoidal = fit_baseline_model(data_MJD,PB,tasc,data_mag,data_err)

    # Sigma clipping
    residuals = data_mag - model
    sigma = np.std(residuals)

    mask = np.abs(residuals) < 3 * sigma

    print(f"Rejected {np.sum(~mask)} of {len(mask)} points.")

    # Apply mask
    data_MJD = data_MJD[mask]
    data_phase = data_phase[mask]
    data_mag = data_mag[mask]
    data_err = data_err[mask]

    # Refit after clipping
    A, B, C, chi2, model, phase, irradiation, ellipsoidal = fit_baseline_model(data_MJD, PB, tasc, data_mag, data_err)


    print(f"The parameters for best baseline model for the {band}-band are:")
    print(f"A = {A:.3f}")
    print(f"B = {B:.3f}")
    print(f"C = {C:.3f}")

    return A, B, C


def calibrate_observation(path, PB, tasc, A_in, B_in, C_in):
    """
    Calibrate one observation run to the common baseline model.

    Input:
    path : path to the reduction file
    PB : orbital period
    tasc : epoch of ascending node
    A_in : irradiation coefficient of the band baseline
    B_in : ellipsoidal coefficient of the band baseline
    C_in : constant offset of the band baseline

    Output:
    observation : observation-run identifier
    shifted_data : dictionary containing calibrated phase, magnitude and uncertainty arrays
    C : fitted constant offset for this observation
    """

    # Get observation name from path
    observation = os.path.basename(os.path.dirname(os.path.dirname(path)))

    data = np.genfromtxt(path)

    MJD = data[:, 0]
    mag = data[:, 2]
    mag_err = data[:, 3]

    def fit_model(MJD, PB, tasc, mag, mag_err, A_in, B_in):
        """
        Fit the baseline model with A and B fixed, leaving only C free.
        """

        phase = ((MJD - tasc) / PB) % 1

        model_AB = (A_in * np.sin(2 * np.pi * phase) + B_in * np.cos(4 * np.pi * phase))

        X = np.ones_like(MJD)[:, None]
        mag_adjusted = mag - model_AB
        X_weight = X / mag_err[:, None]
        mag_weight = mag_adjusted / mag_err
        beta, _, _, _ = np.linalg.lstsq(X_weight, mag_weight, rcond=None)

        C = beta[0]
        model = model_AB + C

        irradiation = A_in * np.sin(2 * np.pi * phase)
        ellipsoidal = B_in * np.cos(4 * np.pi * phase)

        chi2 = np.sum(((mag - model) / mag_err) ** 2)

        return (A_in, B_in,C, chi2, model, phase, irradiation, ellipsoidal)

    # Initial fit
    A, B, C, chi2, model, phase, irradiation, ellipsoidal = fit_model(
        MJD, PB, tasc, mag, mag_err, A_in, B_in)

    # First sigma clip
    residuals = mag - model
    sigma = np.std(residuals)

    mask = np.abs(residuals) < 3 * sigma

    print(f"{observation}: Rejected "
        f"{np.sum(~mask)} of {len(mask)} points.")

    MJD = MJD[mask]
    mag = mag[mask]
    mag_err = mag_err[mask]

    # Second sigma clip
    residuals2 = mag - model[mask]
    sigma2 = np.std(residuals2)

    mask2 = np.abs(residuals2) < 3 * sigma2

    print(f"{observation}: Second round rejected "
        f"{np.sum(~mask2)} of {len(mask2)} points.")

    MJD = MJD[mask2]
    mag = mag[mask2]
    mag_err = mag_err[mask2]

    # Refit after clipping
    A, B, C, chi2, model, phase, irradiation, ellipsoidal = fit_model(
        MJD, PB, tasc, mag, mag_err, A_in, B_in)

    print(f"{observation}: A = {A:.5f}, "
        f"B = {B:.5f}, C = {C:.5f}")

    # Shift dataset to the common baseline
    C_shift = C_in - C
    mag_shift = mag + C_shift

    #shifted_data = {"phase": phase, "mag": mag_shift, "err": mag_err}
    shifted_data = {"phase": phase, "mag": mag_shift, "err": mag_err, "original_mag": mag,"C": C}

    return observation, shifted_data, C


def calibrate_band(paths, PB, tasc, A_in, B_in, C_in):
    '''
    This function calibrates every observation run for a given band

    Inputs:
    paths = paths to reduction files
    PB = orbital period
    tasc = epoch of ascending node
    A_in : irradiation coefficient of the band baseline
    B_in : ellipsoidal coefficient of the band baseline
    C_in : constant offset of the band baseline

    Output:
    calibrated data = dictionary with calibrated runs for each observation run

    '''
    calibrated_data = {}

    for path in paths:
        observation, shifted_data, C = calibrate_observation(path, PB, tasc, A_in, B_in, C_in)
        calibrated_data[observation] = shifted_data

    return calibrated_data

def make_band_paths(file_templates, band):
    """
    Replace {band} in each file path with the requested
    photometric band.
    """

    return [
        path.format(band=band)
        for path in file_templates]

def calibrate_data(file_templates, bands, PB, tasc):
    """
    Calibrate the requested photometric bands.

    Inputs:
    file_templates : Dictionary containing file paths for each band
    bands : bands to calibrate
    PB : orbital period.
    tasc : epoch of ascending node.

    Outputs:
    calibrated : calibrated data for each requested band.
    baseline : dictionary with best-fit baseline parameters (A, B, C) for each band
    """

    calibrated = {}
    baseline = {}

    for band in bands:

        # Create paths for this band
        band_paths = make_band_paths(file_templates, band)

        # Find baseline
        A, B, C = fit_band_baseline(band_paths, PB, tasc, band)

        baseline[band] = (A, B, C)

        # Calibrate all observations
        calibrated[band] = calibrate_band(band_paths, PB, tasc, A, B, C)

    return calibrated, baseline


def plot_calibration_results(calibrated_data, baseline, bands, output_dir):

    """
    Plot the original and calibrated observation runs
    for each band.

    Inputs:
    calibrated_data: calibrated data for each requested band
    baseline: dictionary with best-fit baseline parameters (A, B, C) for each band
    bands: bands to perform calibration on 
    output_dir: name of folder that plots will be placed into 

    Outputs:
    Graph showing each observation run calibrated to the baseline model
    """

    colours = {"g": "green", "r": "red", "i": "gold", "z": "maroon", "u": "blue"}

    for band in bands:

        A_in, B_in, C_in = baseline[band]
        data = calibrated_data[band]

        #one figure, multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        n_eclipse_block = 0
        colour = colours[band]

        for observation, obs_data in tqdm(data.items(), desc=f"Plotting {band}-band calibration"):

            phase = obs_data["phase"]
            mag_shift = obs_data["mag"]
            mag_err = obs_data["err"]
            mag = obs_data["original_mag"]
            C = obs_data["C"]

            phase_grid = np.linspace(np.min(phase), np.max(phase), 1000)
            model_grid = (A_in * np.sin(2 * np.pi * phase_grid)+ B_in * np.cos(4 * np.pi * phase_grid)+ C)
            shift_model = (A_in * np.sin(2 * np.pi * phase_grid) + B_in * np.cos(4 * np.pi * phase_grid)+ C_in)

            # Original data + individual fits
            ax1.errorbar(phase, mag, yerr=mag_err,
                fmt="o", color=colours[band], alpha=0.6,
                zorder=1, capsize=3, linewidth=0.75)

            ax1.plot(phase_grid, model_grid, "--",
                zorder=2, label=f"{observation}: C = {C:.4f}")

            # Shifted/calibrated data + shifted fits
            ax2.errorbar(phase, mag_shift, yerr=mag_err,
                fmt="o", color=colours[band], alpha=0.6,
                zorder=1, capsize=3, linewidth=0.75)

            ax2.plot(phase_grid, shift_model, "--",zorder=3)

            # Eclipse-region plot
            if 0.7 < np.median(phase) < 0.8:

                phase_plot = phase + n_eclipse_block

                ax3.errorbar(phase_plot, mag, yerr=mag_err,
                    fmt="o", color=colours[band], alpha=0.6,
                    zorder=1, capsize=3, linewidth=0.75,
                    label=f"{observation} original")

                ax3.errorbar(phase_plot, mag_shift, yerr=mag_err,
                    fmt="o", color="purple", alpha=0.6,
                    zorder=1, capsize=3, linewidth=0.75,
                    label=f"{observation} calibrated")
                n_eclipse_block += 1

        # Baseline model
        baseline_phases = np.linspace(0, 1, 1000)
        baseline_model = (A_in * np.sin(2 * np.pi * baseline_phases)
            + B_in * np.cos(4 * np.pi * baseline_phases)
            + C_in)

        baseline_phases2 = np.linspace(0, 1 + n_eclipse_block, 1000)

        baseline_model2 = (A_in * np.sin(2 * np.pi * baseline_phases2)
            + B_in * np.cos(4 * np.pi * baseline_phases2)
            + C_in)

        ax1.plot(baseline_phases, baseline_model, color="black", label=f"Baseline: {C_in:.3f}")
        ax2.plot(baseline_phases, baseline_model, color="black", zorder=2)
        ax3.plot(baseline_phases2, baseline_model2, color="black", label=f"Baseline: {C_in:.3f}")

        # Figures
        ax1.set_ylabel("Magnitude")
        ax1.yaxis.set_inverted(True)
        ax1.set_title(f"Fits for {len(data)} {band}-band observation runs")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), ncol=1)

        ax2.yaxis.set_inverted(True)
        ax2.set_ylabel("Magnitude")
        ax2.set_title(f"Shifted Fits for {len(data)} {band}-band observation runs")

        ax3.set_ylabel("Magnitude")
        ax3.invert_yaxis()
        ax3.set_title("Shifted fits for each eclipse observation run")

        fig.supxlabel("Phase")
        fig.suptitle(f"{band}-band calibration", y=0.995)
        fig.subplots_adjust(left=0.09, right=0.78, bottom=0.07, top=0.94, hspace=0.4)
        
        # Save one figure for this band
        filename = os.path.join(output_dir, f"calibration_{band}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close(fig)

#### figure out how to call this

def unpack_shifted_data(shifted_data, A_in, B_in, C_in, phase_min, phase_max):
    '''
    Select the requested phase range from all observations in one band and calculate the baseline residuals.
    
    Inputs:
    shifted_data: calibrated datasets to same offset
    A_in: irradiation coefficient of the band baseline
    B_in: ellipsoidal coefficient of the band baseline
    C_in: constant offset of the band baseline
    phase_min: lower limit for phase binning
    phase_max: upper limit for phase binning

    Outputs:
    phase_to_bin: array of phases for binning
    mag_to_bin: array of magnitudes for binning
    err_to_bin: array of errors for binning
    residuals_bin: array of phases for binning
    '''

    phase_to_bin = []
    mag_to_bin = []
    err_to_bin = []

    for observation, data in shifted_data.items():

        # print(observation)
        phase = data["phase"]
        mag = data["mag"]
        err = data["err"]

        # Select desired phase range
        mask = (phase >= phase_min) & (phase <= phase_max)
        phase_section = phase[mask] % 1
        mag_section = mag[mask]
        err_section = err[mask]

        phase_to_bin += list(phase_section)
        mag_to_bin += list(mag_section)
        err_to_bin += list(err_section)

    phase_to_bin = np.array(phase_to_bin)
    mag_to_bin = np.array(mag_to_bin)
    err_to_bin = np.array(err_to_bin)

    # Model evaluated at the selected phase points
    model_bin = (A_in * np.sin(2 * np.pi * phase_to_bin)
        + B_in * np.cos(4 * np.pi * phase_to_bin) + C_in)

    residuals_bin = model_bin - mag_to_bin

    return (phase_to_bin, mag_to_bin, err_to_bin, residuals_bin)

def create_the_beast(calibrated, baseline, phase_min, phase_max, bands):
    """
    Combine the calibrated data from the requested bands within the selected phase range.
    
    Inputs:
    calibrated: output of function calibrate data, calibrated data for each requested band
    baseline: output of function calibrate data, dictionary with best-fit baseline parameters (A, B, C) for each band
    phase_min: lower limit for phase binning
    phase_max: upper limit for phase binning
    bands: bands used
    
    Outputs:
    beast_phases: phases
    beast_mag: magnitudes
    beast_err: magnitude errors
    beast_resids: residuals
    band_lims: limits
    band_OBs: observing block for bands
    """

    beast_phases = []
    beast_mag = []
    beast_resids = []
    beast_err = []

    band_lims = []
    band_OBs = []

    for band in bands:

        # Get calibrated data and baseline for this band
        data = calibrated[band]
        A_in, B_in, C_in = baseline[band]

        # Get data for this band
        phase, mag, err, residuals = unpack_shifted_data(data, A_in, B_in, C_in,
                                                        phase_min, phase_max)

        band_OBs.append(band)

        # Sort by phase
        sort_idx = np.argsort(phase)

        phase = phase[sort_idx]
        mag = mag[sort_idx]
        err = err[sort_idx]
        residuals = residuals[sort_idx]

        beast_phases += list(phase)
        beast_mag += list(mag)
        beast_err += list(err)
        beast_resids += list(residuals)

        band_lims.append(len(phase))

    # print(np.array(beast_phases))
    # print(np.array(beast_phases).shape[0])
    # print(band_lims)

    return (np.array(beast_phases), np.array(beast_mag), np.array(beast_err),
            np.array(beast_resids), band_lims, band_OBs)

def create_boxcar(phases, centre, width):
    '''
    Creates a simple boxcar at a desired phase centre and phase width

    Input:
    Phases: phase range
    Centre: phase centre
    Width: width of boxcar

    Output:
    boxcar: simple boxcar function
    '''

    boxcar = np.zeros_like(phases)
    mask = np.abs(phases - centre) <= (width/2)
    boxcar[mask] = 1

    return boxcar

def define_matrix(centre, width, phases, limits):
    '''
    Defines a matrix that will be used to detrend the data

    Input:
    Phases: phase range
    Centre: phase centre
    Width: width of boxcar
    Limits: limits for the matrix 

    Output:
    matrix: a matrix of desired size
    '''

    matrix = np.zeros((len(phases), 1 + 3 * len(limits)))
    low_lim = 0
    for i, l in enumerate(limits):
            band_phases = phases[low_lim:l+low_lim]
            boxcar = create_boxcar(band_phases, centre, width)
            matrix[low_lim:l+low_lim,0] = boxcar
            matrix[low_lim:l+low_lim, 1+i] = np.ones_like(band_phases)
            x =  band_phases - centre
            matrix[low_lim:l+low_lim, 1 + len(limits) + i] = np.zeros_like(band_phases) + x
            matrix[low_lim:l+low_lim, 1 + 2 * len(limits) + i] = np.zeros_like(band_phases) + x**2

            low_lim = l + low_lim
    return matrix

def search_boxcar(phases, resids, err, limits, phase_min, phase_max, min_width, max_width, n_phase=100, n_width=100):
    """
    Search for the boxcar that maximises the amplitude/error score.

    Inputs:
    phases: phase range 
    resids: shifted magnitude residuals 
    err: magnitude errors
    limits: limits for search
    phase_min: minimum phase
    phase_max: maxmimum phase
    min_width: minimum width of boxcar
    max_width: maximum width of boxace
    n_phase: grid for phase search dimension (set to 100)
    n_width: grid for width search dimension (set to 100)

    Outputs
    dictionary of searched phases and searched widths with the corresponding amplitudes, uncert, coeffs,
    scores and the best id, best phase and best width found
    """

    weights = 1 / err**2
    search_phases = np.linspace(phase_min, phase_max, n_phase)
    search_widths = np.linspace(min_width, max_width, n_width)

    amplitudes = np.zeros((n_phase, n_width))
    uncerts = np.zeros((n_phase, n_width))
    coeffs = np.zeros((n_phase, n_width, 1 + 3 * len(limits)))

    with tqdm (total = n_phase*n_width, desc = "Searching boxcar grid") as pbar:
        for i, c in enumerate(search_phases):
            for j, w in enumerate(search_widths):
                #print(f"Getting matrix for {c}, {w}")

                m = define_matrix(c, w, phases, limits)
                XTWX = (m.T * weights) @ m
                XTWy = m.T @ (weights * resids)
                C = np.linalg.solve(XTWX,XTWy)
                coeffs[i, j, :] = C
                covariance = np.linalg.inv(XTWX)
                amplitude = C[0]
                amplitude_err = np.sqrt(covariance[0, 0])
                amplitudes[i, j] = amplitude
                uncerts[i, j] = amplitude_err
            
                pbar.update(1)

    scores = amplitudes / uncerts

    best_idx = np.unravel_index(np.nanargmax(scores), scores.shape)
    best_phase = search_phases[best_idx[0]]
    best_width = search_widths[best_idx[1]]
    best_score = scores[best_idx]
    best_coeffs = coeffs[best_idx[0], best_idx[1],:]
    best_amplitude = best_coeffs[0]

    print(f"Maximum score = {best_score}")
    print(f"Best centre = {best_phase}")
    print(f"Best width = {best_width}")
    print(f"Best amplitude = {best_amplitude}")

    #return a dict for easy access to variables
    return {"search_phases": search_phases,
        "search_widths": search_widths,
        "amplitudes": amplitudes,
        "uncerts": uncerts,
        "scores": scores,
        "coeffs": coeffs,
        "best_idx": best_idx,
        "best_phase": best_phase,
        "best_width": best_width,
        "best_score": best_score,
        "best_coeffs": best_coeffs,
        "best_amplitude": best_amplitude}


def plot_boxcar_surface(result, output_dir):
    """
    Plot the boxcar correlation surface and its centre/width traces.

    Input:
    result: dictionary containing results from search_boxcar 
    output_dir: folder where plots will be placed

    Output:
    corrrealtion surface plot
    """

    #reading the dictionary
    search_phases = result["search_phases"]
    search_widths = result["search_widths"]
    scores = result["scores"]
    best_idx = result["best_idx"]
    best_phase = result["best_phase"]
    best_width = result["best_width"]

    fig = plt.figure(figsize=(10, 8))

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1],
                            hspace=0.1, wspace=0.1)

    ax_main = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 0], sharey=ax_main)
    contour = ax_main.contourf(search_phases, search_widths, scores.T, levels=100)
    ax_main.scatter(best_phase, best_width, color="black", marker="x", s=100, label="Best match")

    ax_main.set_title("Boxcar correlation surface")

    # Maximum score as a function of centre
    width_trace = np.max(scores, axis=1)

    ax_bottom.plot(search_phases, scores[:, best_idx[1]], "--")
    ax_bottom.plot(search_phases, width_trace, "--")
    ax_bottom.set_xlabel("Boxcar centre phases")
    ax_bottom.set_ylabel("Correlation")

    # Maximum score as a function of width
    centre_trace = np.max(scores,axis=0)
    ax_side.plot(scores[best_idx[0], :], search_widths,"--")
    ax_side.plot(centre_trace, search_widths, "--")
    ax_side.set_xlabel("Correlation")
    ax_side.set_ylabel("Boxcar width")
    ax_side.invert_xaxis()

    plt.setp(ax_main.get_xticklabels(),visible=False)
    plt.setp(ax_main.get_yticklabels(),visible=False)

    cbar = fig.colorbar(contour,ax=[ax_main, ax_bottom, ax_side],
        label="Combined correlation")

    plt.show()
    filename = os.path.join(output_dir, f"correlation_surface.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    

def plot_combined_boxcar(phases, resids, err, result, phase_min, phase_max, output_dir):
    
    """
    Plot all residuals together with the best-fitting boxcar

    Inputs:
    phases: phase array
    resids: magnitude residuals
    err: magnitude errors
    result: dictionary
    phase_min: mimnum phase
    phase_max: maximum phase
    output_dir: folder name where plot is saved

    Outputs:
    Figure with all bands plotted together with best boxcar
    """

    best_phase = result["best_phase"]
    best_width = result["best_width"]
    best_score = result["best_score"]
    best_amplitude = result["best_amplitude"]

    phase_grid = np.linspace(phase_min, phase_max,1000)

    boxcar_best = create_boxcar(phase_grid, best_phase,best_width)

    plt.figure(figsize=(10, 5))
    plt.errorbar(phases, resids, yerr=err,fmt="o", color="hotpink", alpha=0.6, capsize=3,
        linewidth=0.75, label="Residuals", zorder=1)
    plt.plot(phase_grid, best_amplitude * boxcar_best,
        label="boxcar function",zorder=2)
    plt.fill_between(phase_grid, best_amplitude * boxcar_best,
        color="lightblue", alpha=0.3, zorder=2)

    plt.legend()
    plt.xlabel("Phase")
    plt.ylabel("Residuals")

    plt.title(f"Combined best boxcar: "
        f"center = {best_phase:.3f}, "
        f"width = {best_width:.3f}, "
        f"score = {best_score:.3f}, "
        f"amplitude = {best_amplitude:.5f}")

    plt.show()
    filename = os.path.join(output_dir, f"boxcar_all_bands.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def bin_residuals(phases, resids, err, n_bins, phase_min, phase_max):
    
    """
    Calculate weighted binned residuals.

    Inputs:
    phases = phase array
    resids = magnitdue residuals array
    err = magnitude errors
    n_bins = number of bins in plot 
    phase_min = min phase for plot
    phase_max = max phase for plot

    Outputs:
    phase_bin = binned phase array 
    resid_bin = binned residuals array
    err_bin = binned error array 
    """

    phase_bin = []
    resid_bin = []
    err_bin = []

    edges = np.linspace(phase_min, phase_max,n_bins + 1)

    for j in range(n_bins):
        mask_bin = ((phases >= edges[j]) & (phases < edges[j + 1]))

        if np.any(mask_bin):
            weight = 1 / err[mask_bin]**2
            phase_bin.append(np.average(phases[mask_bin],weights=weight))
            resid_bin.append(np.average(resids[mask_bin],weights=weight))
            err_bin.append(np.sqrt(1 / np.sum(weight)))

    return (np.array(phase_bin), np.array(resid_bin), np.array(err_bin))

def plot_band_residuals(phases, resids, err, limits, bands, result, phase_min, phase_max, output_dir, n_bins=50):
    """
    Plot the residuals separately for each band,
    including the best-fitting boxcar and detrending terms.

    Input:
    phases: phase array
    resids: magnitude residuals
    err: magnitude errors
    limits: binning limits
    bands: bands to include on plot
    result: dictionary
    phase_min: mimnum phase
    phase_max: maximum phase
    output_dir: folder name where plot is saved
    n_bins: number of bins for the plot 

    Output:
    Plot showing magnitude residuals VS phase for each requested band together with the best boxcar to show 
    the found phase range for the gravitational lensing feature and its amplitude. The scorecan be regarded as a SNR

    """

    colours = {"u": "navy",
        "g": "forestgreen",
        "r": "orange",
        "i": "red",
        "z": "maroon"}

    best_idx = result["best_idx"]
    best_coeffs = result["best_coeffs"]
    best_phase = result["best_phase"]
    best_width = result["best_width"]
    best_score = result["best_score"]
    best_amplitude = result["best_amplitude"]

    fig = plt.figure(figsize = (10,8))
    grd = fig.add_gridspec(len(bands),1,hspace=0.15)

    low_lim = 0

    plot_matrix = define_matrix(best_phase, best_width, phases, limits)

    axes = []

    for i, band in enumerate(bands):
        axes.append(fig.add_subplot(grd[i, 0]))
        ax = axes[-1]
        up_lim = low_lim + limits[i]
        plot_phases = phases[low_lim:up_lim]
        plot_resids = resids[low_lim:up_lim]
        plot_err = err[low_lim:up_lim]

        # Weighted binning
        phase_bin, resid_bin, err_bin = bin_residuals(plot_phases, plot_resids, plot_err, n_bins, phase_min, phase_max)

        # Design-matrix terms
        boxcar = plot_matrix[low_lim:up_lim,0]
        linear = plot_matrix[low_lim:up_lim, 1 + i + len(bands)]
        quadratic_best = plot_matrix[low_lim:up_lim, 2 * len(bands) + i + 1]

        coeff_offset = best_coeffs[1 + i]
        coeff_linear = best_coeffs[1 + i + len(bands)]
        coeff_quad = best_coeffs[2 * len(bands) + i + 1]

        offset = (np.ones_like(plot_phases)* coeff_offset)
        plot_linear = (coeff_linear * linear)
        plot_quad = (coeff_quad * quadratic_best)
        detrend = (boxcar * best_amplitude+ plot_linear+ plot_quad)

        chi2 = np.sum(((plot_resids - offset - detrend)/ plot_err)**2)
        reduced_chi2 = (chi2/ (plot_phases.shape[0] - 4))

        ax.errorbar(phase_bin, resid_bin- np.ones_like(phase_bin) * coeff_offset, err_bin,
            color=colours[band],linestyle="none", alpha=0.5)
        ax.plot(plot_phases,detrend,color="k")
        ax.text(0.03, 0.78, f"Reduced $\\chi^2$ = {reduced_chi2:.2f}", transform=ax.transAxes)
        ax.set_xlim(phase_min,phase_max)

        if i != len(bands) - 1:
            ax.tick_params(bottom=False, labelbottom=False)

        ax.set_ylabel(f"{band}$_s$ resids (mags)")
        low_lim = up_lim

    axes[0].set_title(f"Center = {best_phase:.3f}, "
        f"width = {best_width:.3f}, "
        f"score = {best_score:.3f}, "
        f"amplitude: {best_amplitude:.5f}")

    axes[-1].set_xlabel("Phase")
    plt.show()
    filename = os.path.join(output_dir, "GL_lensing_bands.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def run_analysis(config):
    """
    This function calls functions in this .py file and uses info from the user input config file
    """

    os.makedirs(config["output_dir"], exist_ok=True)

    calibrated, baseline = calibrate_data(config["file_templates"], config["bands"], config["PB"], config["tasc"])
    plot_calibration_results(calibrated, baseline, config["bands"], config["output_dir"])
    phases, mag, err, resids, limits, band_OBs = create_the_beast(calibrated, baseline, config["phase_min"], config["phase_max"], config["bands"])
    result = search_boxcar(phases, resids, err,limits,
                        config["phase_min"],config["phase_max"], config["min_width"], config["max_width"],
                        config["n_phase"],config["n_width"])

    plot_boxcar_surface(result, config["output_dir"])
    plot_combined_boxcar(phases, resids, err, result, config["phase_min"], config["phase_max"], config["output_dir"])
    plot_band_residuals(phases, resids, err, limits, config["bands"], result, config["phase_min"],config["phase_max"], config["output_dir"])

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Usage: python GL_binning.py config.txt")

    config = read_config(sys.argv[1])
    run_analysis(config)

# # TESTING
# #check results for now by running:

# # Paths to the reduction files, with {band} as a placeholder
# file_templates = [
#     'OB0002/Reduction files/FINALrun0008_{band}.dat',
#     'OB0003/Reduction_files/FINALrun0009_{band}.dat',
#     'OB0004/Reduction_files/FINALrun0010_{band}.dat',
#     'OB0005/Reduction_files/FINALrun0008_{band}.dat',
#     'OB0005_2/Reduction_files/FINALrun0008_{band}.dat',
#     'OB0008/Reduction_files/FINALrun0007_{band}.dat',
#     'OB0009/Reduction_files/FINALrun0009_{band}.dat',
#     'OB0010/Reduction_files/FINALrun0010_{band}.dat',
#     'OB0029/Reduction_files/FINALrun0011_{band}.dat',
#     'OB0030/Reduction_files/FINALrun0012_{band}.dat',
#     'OB0001/Reduction_files/FINALrun0007_{band}.dat',
#     'OB0006/Reduction_files/FINALrun0009_{band}.dat',
#     'OB0007/Reduction_files/FINALrun0010_{band}.dat',
#     'OB0011/Reduction_files/FINALrun0011_{band}.dat',
#     'OB0012/Reduction_files/FINALrun0012_{band}.dat',
#     'OB0014/Reduction_files/FINALrun0009_{band}.dat',
#     'OB0013/Reduction_files/FINALrun0013_{band}.dat',
#     'OB0016/Reduction_files/FINALrun0007_{band}.dat',
#     'OB0015/Reduction_files/FINALrun0013_{band}.dat',
#     'OB0017/Reduction_files/FINALrun0007_{band}.dat',
#     'OB0018/Reduction_files/FINALrun0007_{band}.dat',
#     'OB0019/Reduction_files/FINALrun0006_{band}.dat',
#     'OB0020/Reduction_files/FINALrun0020_{band}.dat',
#     'OB0021/Reduction_files/FINALrun0020_{band}.dat']

# for band in ["u", "g", "r", "i", "z"]:
#     band_paths = make_band_paths(file_templates, band)

#     print("\n", band)
#     print(band_paths[0])

# bands = ["u", "g", "r", "i", "z"]


# calibrated, baseline = calibrate_data(
#     file_templates,
#     bands=bands,
#     PB=0.3608934817,
#     tasc=56047.5490643)

# output_dir = "GL_binning_plots"

# os.makedirs(output_dir, exist_ok=True)

# plot_calibration_results(
#     calibrated,
#     baseline,
#     bands,
#     output_dir
# )



# phase_min = 0.72
# phase_max = 0.78

# phases, mag, err, resids, limits, band_OBs = create_the_beast(
#     calibrated,
#     baseline,
#     phase_min,
#     phase_max,
#     bands)
    

# min_width = 0.01
# max_width = 0.02

# result= search_boxcar(phases,
#     resids,
#     err,
#     limits,
#     phase_min,
#     phase_max,
#     min_width,
#     max_width)

# plot_boxcar_surface(result, output_dir)

# plot_combined_boxcar(
#     phases,
#     resids,
#     err,
#     result,
#     phase_min,
#     phase_max, output_dir
# )

# plot_band_residuals(
#     phases,
#     resids,
#     err,
#     limits,
#     bands,
#     result,
#     phase_min,
#     phase_max, output_dir
# )

