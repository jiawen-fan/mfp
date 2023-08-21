from scipy import integrate
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import pyplot as plt      
import pandas as pd
from scipy import special                 
import array
import scipy as sp
import scipy.interpolate
import re
import pickle as pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import concurrent
def unPackRawFile(raw_path):
    """
    - unpacks the .raw file. Not used for the neural network.
    """
    y = np.loadtxt(raw_path, skiprows=6)
    distance = y[:,0]
    pec_vel = y[:,1]
    temp = y[:,2]
    HI_density = y[:,3]
    gas_density = y[:,4]
    gas_metallicity = y[:,5]
   
    return distance, pec_vel, temp, HI_density, gas_density, gas_metallicity

def getPos(path_LOS,linenumber=8):
    """
    the start position of the LOS is given inside each file, (in the comments)
    this function parses the comments to get that information
    """
    f = open(path_LOS)
    x = f.readlines()[linenumber]
    answer = re.search('\(([^)]+)', x).group(1)
    arr = np.array(answer.split(','),dtype=float)
    return arr

def getDir(path_LOS,linenumber=8):
    """
    the direction of the LOS is given inside each file, (in the comments)
    this function parses the comments to get that information
    """
    f = open(path_LOS)
    x = f.readlines()[linenumber]
    answer = re.search('\(([^)]+)', x.split(', ')[1]).group(1)
    arr = np.array(answer.split(','),dtype=float)
    return arr


def convertSphereToCart(theta, phi):
    "converts a unit vector in spherical to cartesian, needed for getGalaxies"
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
def change_to_redshift(distance,redshift):
    '''change Cmpc to redshift, start at the redshift of the box'''
    distance_redshift = []
    distance_redshift.append(redshift)
    distance_difference = redshift
    for i in range(len(distance)-1):
        distance_difference -= (distance[i+1] - distance[i]) / 100 * .23
        distance_redshift.append(distance_difference)
    return distance_redshift

def change_to_wavelength(redshift_distance,redshift):
    start_wavelength = 912
    wavelength = []
    for i in range(len(redshift_distance)):
        wavelength.append(912*(1+redshift_distance[i])/(1+redshift))
    return wavelength

def low_region(redshift,number):
       """
       finds the low neutral fraction area. Calculate the optical depth until it's unity. Return the average length
       """
       sigma_value = 6.3e-18
       neutral_fraction_limit = 1e-3
       mfp = []
       for i in range(number,number+40):
            mean_free_path_temp = 0
            raw_path = 'los.00' + '{0:03}'.format(i) +'.raw'
            distance, pec_vel, temp, HI_density, gas_density, gas_metallicity = unPackRawFile(raw_path)
            distance_redshift = change_to_redshift(distance,redshift)
            '''please look over the sigma change'''
            #changing the distance from chimp to cemeters
            factor = 1/(redshift+1)*3.086e24 /0.68
            distance_cm = distance *factor
            k = 1
            wavelength = change_to_wavelength(distance_redshift,redshift)
            while k< len(distance) -2:
                total_optical_depth = [0]
                mfp_temp = 0
                distance_graph = []
                HI_graph = []
                optical_depth_stuff = []
                sigma = []
                while(k< len(distance) -2):
                    distance_graph.append(distance_cm[k])
                    if(HI_density[k]/gas_density[k]<=1e-3):
                        HI_graph.append(HI_density[k])
                    else:
                        HI_graph.append(7.7e-6)
                    sigma.append(sigma_value)
                    total_optical_depth = integrate.cumtrapz(np.multiply(HI_graph,sigma),distance_graph,initial =0)
                    optical_depth_stuff.append(total_optical_depth[-1])
                    mfp_temp += distance[k] - distance[k-1]
                    k+=1
                x = []
                for p in wavelength[3:]:
                    x.append(p)
                mfp.append([x,optical_depth_stuff])
       return mfp

def binning(mfp,redshift):
    raw_path = 'los.00' + '{0:03}'.format(0) +'.raw'
    distance, pec_vel, temp, HI_density, gas_density, gas_metallicity = unPackRawFile(raw_path)
    distance_redshift = change_to_redshift(distance,redshift)
    wavelength = change_to_wavelength(distance_redshift,redshift)
    line = np.linspace(wavelength[0],wavelength[-1],1500)
    every_bins = []
    for z in (mfp):
        value_bins = []
        k = 0
        for i in range(len(line)-2):
            counts = 0
            value = 0
            while( line[i+1] < z[0][k] < line[i]):
                value += z[1][k]
                k+=1
                counts +=1
            if(counts == 0):
                value_bins.append(0)
            else:
                value_bins.append(value/counts)
        every_bins.append(value_bins)
    return every_bins

def average_of_bins(bins_values):
    average = []
    for i in range(len(bins_values[0])):
        total_value = 0
        for k in range(len(bins_values)):
            total_value += bins_values[k][i]
        average.append(total_value/len(bins_values[0]))
        total_value = 0
    return average


def median_of_bins(bins_values):
    median = []
    for i in range(len(bins_values[0])):
        total_value = []
        for k in range(len(bins_values)):
            total_value.append(bins_values[k][i])
        median.append(np.median(total_value))
        total_value = 0
    return median

def main(space):
    redshift = 5.88
    result = []
    #spacing for los spacing
    #line for bin distance
    raw_path = 'los.00' + '{0:03}'.format(0) +'.raw'
    distance, pec_vel, temp, HI_density, gas_density, gas_metallicity = unPackRawFile(raw_path)
    distance_redshift = change_to_redshift(distance,redshift)
    wavelength = change_to_wavelength(distance_redshift,redshift)
    line = np.linspace(wavelength[0],wavelength[-1],1500)
    mfp = low_region(redshift= redshift, number=int(space))
    bins_values = binning(mfp = mfp,redshift=redshift)
    average = average_of_bins(bins_values= bins_values)
    median = median_of_bins(bins_values = bins_values)
    result.append([average,median])
    return(result)

# process = np.linspace(0,400,11)
# results = []
# for i in range(10):
#     result = multiprocessing.Process(target = main(5.88,process[i]))
#     result.start()
#     results.append(result)

# for result in results:
#     result.join()

with ProcessPoolExecutor() as executor:
    process = np.linspace(0,400,11)
    results = executor.map(main,process)
final = []
for i in results:
    final.append(i)
with open('results.pkl', 'wb') as f:
    pickle.dump(final, f, protocol=pickle.HIGHEST_PROTOCOL)