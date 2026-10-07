import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import os


def convert_P1_to_super (csv_mags):
    '''
    This function takes an input of pan-STARRS P1 magnitudes as a csv file and uses the 
    linear model coefficients from JL Tonry et al. 2012 to convert to super SDSS magnitudes
    '''

    data = np.genfromtxt(csv_mags, delimiter = ",", missing_values = "nan", skip_header = 1)
    apnum = data[:,0]
    gp1 = data[:,1]
    rp1 = data[:,2]
    ip1 = data[:,3]
    zp1 = data[:,4]

    #first we convert to SDSS'
    gSDSS_results = np.array([])
    rSDSS_results = np.array([])
    iSDSS_results = np.array([])
    zSDSS_results = np.array([])

    for i in range (len(apnum)):

        #computes color
        diff = gp1[i] - rp1[i] 

        #g-band to SDSS
        gSDSS = ((0.162*diff)+0.014) + gp1[i]
        gSDSS_results = np.append(gSDSS_results, gSDSS)

        #r-band to SDSS
        rSDSS = ((0.0011*diff)-0.001) + rp1[i]
        rSDSS_results = np.append(rSDSS_results, rSDSS)

        #r-band to SDSS
        iSDSS = ((0.02*diff)-0.004) + ip1[i]
        iSDSS_results = np.append(iSDSS_results, iSDSS)

        #r-band to SDSS
        zSDSS = ((-0.05*diff)+0.013) + zp1[i]
        zSDSS_results = np.append(zSDSS_results, zSDSS)

        #print (f"aper{apnum[i]} {gSDSS:.5f} {rSDSS:.5f} {iSDSS:.5f} {zSDSS:.5f}")

    for i in range (len(apnum)):

        #computes color
        diff_gr = gSDSS_results[i] - rSDSS_results[i] 
        diff_ri = rSDSS_results[i] - iSDSS_results[i]

        if diff_gr < 1.3:
            print ("Accepted magnitude range")

        else:
            print (f"WARNING: Magnitude of aper{apnum[i]} out of range but will proceed with conversion anyway")


        #g-band super SDSS
        gsuper = ((-0.047*diff_gr)-0.009) + gSDSS_results[i]

        #r-band super SDSS
        rsuper = ((-0.004*diff_gr)) + rSDSS_results[i]

        #i-band super SDSS
        isuper = ((-0.093*diff_ri) + 0.005) + iSDSS_results[i]

        #z-band super SDSS
        zsuper = zSDSS_results[i]

        print (f"aper{apnum[i]} {gsuper:.5f} {rsuper:.5f} {isuper:.5f} {zsuper:.5f}")

import sys
convert_P1_to_super(sys.argv[1])