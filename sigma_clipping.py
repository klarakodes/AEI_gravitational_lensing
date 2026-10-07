import numpy as np 
import os 
import glob
import matplotlib.pyplot as plt

def clipping(folder):

    PB = 0.3608934817
    tasc = 56047.5490643

    paths = [path for path in glob.glob(os.path.join(folder, "*.dat"))
            if not os.path.basename(path).startswith("FINAL")]

    def fit_model(MJD, PB, tasc, mag, mag_err, A_in, B_in):
            '''
            Fit A*sin(2*pi*(t-t_asc)/P_B) + B*cos(4*pi*(t-t_asc)/P_B) + C for a fixed period P_B and T_asc.
            '''

            # compute phase
            phase = ((MJD - tasc) / PB) % 1

            model_AB = (A_in * np.sin(2*np.pi*phase) + B_in * np.cos(4*np.pi*phase))
            

            # create matrix
            X = np.ones_like(MJD)[:, None]
            mag_adjusted = mag - model_AB

            # weighted least squares
            X_weight = X / mag_err[:, None]
            mag_weight = mag_adjusted / mag_err

            beta, _, _, _ = np.linalg.lstsq(X_weight, mag_weight, rcond=None)

            C = beta[0]

            model = model_AB + C

            irradiation = A_in * np.sin(2*np.pi*phase)
            ellipsoidal = B_in * np.cos(4*np.pi*phase)

            chi2 = np.sum(((mag - model) / mag_err)**2)

            return A_in, B_in, C, chi2, model, phase, irradiation, ellipsoidal

    #band fitting paramaters
    band_params = {"g": (0.019, -0.037, 18.027),
                   "r": (0.017, -0.036, 18.315),
                   "i": (0.016, -0.034, 18.610),
                   "z": (0.011, -0.03, 18.844),
                   "u": (0.023, -0.048, 17.756)}

    for i, path in enumerate(paths):

        filename = os.path.basename((path))
        band = filename[-5]
        A_in, B_in, C_in = band_params[band]
        data = np.genfromtxt(path)
            
        print (f"Processing {filename} {band} - band")

        #load data
        MJD = data[:, 0]
        data_phase = data[:, 1]
        mag = data[:, 2]
        mag_err = data[:, 3]

        # Keep original data for sanity-check plot
        MJD_original = MJD.copy()
        phase_original = data_phase.copy()
        mag_original = mag.copy()


        # Initial fit
        A, B, C, chi2, model, phase, irradiation, ellipsoidal = fit_model(MJD, PB, tasc, mag, mag_err, A_in, B_in)

        # Clip outliers 
        while True: 
            residuals = mag - model
            sigma = np.std(residuals)
            mask = np.abs(residuals) < 3 * sigma
            n_rejected = np.sum(~mask)
            print(f"Dataset {i+1}: Rejected {n_rejected} of {len(mag)} points.")

            if n_rejected == 0:
                break

            MJD = MJD[mask]
            data_phase = data_phase[mask]
            mag = mag[mask]
            mag_err = mag_err[mask]
            model = model[mask]

        print(f"Clipping done! Final points {len(mag)} out of {len(mag_original)}")

        #sanity check figure
        plt.figure(figsize=(10, 5))
        phase_plot = np.linspace(0, 1, 500)
        model_plot = (A_in * np.sin(2*np.pi*phase_plot) + B_in * np.cos(4*np.pi*phase_plot) + C)
        plt.scatter(phase_original, mag_original, s=8, alpha=0.4, label="Original")  # Original data
        plt.scatter(data_phase, mag, s=8,alpha=0.8,label="After 3σ clipping") # Clipped/retained data
        plt.plot(phase_plot, model_plot, color = "black",label="Baseline model")

        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        plt.title(f"{filename} — {band}-band sigma clipping")
        plt.legend()
        plt.tight_layout()

        # Save plot in same folder
        plot_path = os.path.join(os.path.dirname(path), f"CLIPPING_{filename.replace('.dat', '.png')}")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"Saved clipping plot: {plot_path}")
            
        # Save cleaned data
        output_path = os.path.join(os.path.dirname(path), "FINAL" + filename)
        output_data = np.column_stack([MJD, data_phase, mag, mag_err])
        np.savetxt(output_path, output_data)
        print(f"Saved: {output_path}")


import sys
clipping(sys.argv[1])