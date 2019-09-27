#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept 27 15:13 2019
This code reproduces Figure 7 (Figure 8 can also be plotted by changing two lines of this code) of the following paper:
    - Viens L. and Van Houtte C., Denoising ambient seismic field correlation functions with convolutional autoencoders (submitted to GJI) 
The code computes relative velocity changes (dv/v) from single-station correlation (SC) functions calculated at one MeSO-net station in the Tokyo metropolitan area (e.g., the NS7M station).
dv/v measurements from 3 types of SC functions are calculated:
    1) dv/v from raw SC functions
    2) dv/v from SC functions denoised with the SVDWF method from Moreau et al., (2017, GJI)
    3) dv/v from SC functions denoised with ConvDeNoise, which is the convolutional denoising autoencoder presented in the paper.

The weigths of ConvDeNoise used in this example have been obtained following the procedure detailed in the paper and the ConvDeNoise algorithm can be found at: https://github.com/lviens 
We only focus on a short time period in this example (e.g., 16 days) as the entire SC function file is too large to be uploaded on Github. 

@author: Loic Viens
Questions/Comments -> loicviens@gmail.com

################################################   IMPORTANT  ################################################

                                        The code uses Keras version 2.2.4
                                        
################################################   IMPORTANT  ################################################   
  
"""

#%%


from __future__ import division

import sys
import numpy as np
from keras.models import load_model

import scipy.io as sio
from scipy import signal
import pandas as pd
import datetime 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

from functions_for_autoencoders import  stretching_current

#%% Load SC functions and reference waveforms
"""
The Test_data.mat matrix contains a dictionary for the data at the NS7M station with:
    - 'SC': the RAW 20-min SC functions for the Z-N and Z-E components from 2017/09/25 to 2017/10/11: (1152, 200, 2) -> (16days @ 20 min, 1 s signal with a 200 Hz sampling, 2 components)
          The SC functions have been filtered between 1 and 20 Hz with a Butterworth filter, tapered, and are normalized between 0 and 1
          
    - 'SC_WIE_NS': 20-min Z-N SC functions after denoising with the SVDWF method from 2017/09/25 to 2017/10/11: (1152, 200) -> (16days every 20 min, 1 s signal with a 200 Hz sampling)
                   Raw Z-N SC functions have been filtered between 1 and 20 Hz with a Butterworth filter, tapered, and the SVDWF was applied (25 singular values, K = 5, L = 5)
    - 'SC_WIE_EW': 20-min Z-E SC functions after denoising with the SVDWF method from 2017/09/25 to 2017/10/11: (1152, 200) -> (16days every 20 min, 1 s signal with a 200 Hz sampling)
                   Raw Z-E SC functions have been filtered between 1 and 20 Hz with a Butterworth filter, tapered, and the SVDWF was applied (25 singular values, K = 5, L = 5)
          
    - 'raw_ref_NS':  Reference waveform the Z-N raw SC functions (stack from April 1 to December 31, 2017)
    - 'raw_ref_EW':  Reference waveform the Z-E raw SC functions (stack from April 1 to December 31, 2017)
    - 'Wie_ref_NS':  Reference waveform the Z-N SC functions denoised with the SVDWF method (stack from April 1 to December 31, 2017)
    - 'Wie_ref_EW':  Reference waveform the Z-E SC functions denoised with the SVDWF method (stack from April 1 to December 31, 2017)
    - 'Conv_ref_NS': Reference waveform the Z-N SC functions denoised with ConvDeNoise (stack from April 1 to December 31, 2017)
    - 'Conv_ref_EW': Reference waveform the Z-E SC functions denoised with ConvDeNoise (stack from April 1 to December 31, 2017)  
    
    - 'Precip':      20-min precipitation at the Funabashi weather station (~5km away from NS7M) from 2017/09/25 to 2017/10/11
    - 'Temp':        20-min temperature at the Funabashi weather station (~5km away from NS7M) from 2017/09/25 to 2017/10/11
    
"""


#%% Change the fig_choice
fig_choice = 7 # EITHER 7 or 8

if fig_choice== 7 or  fig_choice == 8:
    print('Plotting Figure ' + str(fig_choice) )
else:
    print(   '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    sys.exit('The fig_choice variable has to be equal to either 7 or 8')
    
#%% Load data

dir_ini = './'
input_data = dir_ini + 'Test_data_fin.mat'
data = sio.loadmat(input_data)
SCfunc = data['SC']

#%% Load the weights of ConvDeNoise and denoise the SC functions
autoencoder = load_model(dir_ini + '/ConvDeNoise_NS7M_station.h5')
Denoised_SCF = autoencoder.predict(SCfunc)

#%%
hann = signal.hann(11) # Hanning function to taper the SC functions
delta = 200 # Sampling rate
t_vec = np.arange(0, 1, 1/delta)

Epsilon = .05 # Stretching between -Epsilon to +Epsilon (multiply by 100 to get the dv in %) STRETCHING LIMITS
nbtrial = 20 # Number of tries for the first determination of dv (2 steps see paper)
t_ini = 0  # Time to start computing the dv/v (in second). Zero is at the zero lag time
t_length = 1 # Length of the signal over which the dv/v is computed (in second)
t_ini_d = t_ini*delta          # start dv/v computation at t_ini_d/delta seconds from the signal begining
t_length_d = int(t_length*delta)


window_stretch = np.arange(t_ini,int(t_length_d))

#%% Compute dv/v from raw SC functions (Pre-processing: removing the mean of each SC function before performing the stretching)   
dvxffin = np.empty((2,int(len(SCfunc))))
ccxffin = np.empty((2,int(len(SCfunc)))) # Set variables
errorxffin = np.empty((2,int(len(SCfunc))))
for ii in [0, 1]:
    if ii== 0:
        refd = np.squeeze(data['raw_ref_NS'])
    elif ii== 1:
        refd = np.squeeze(data['raw_ref_EW'])
    xffin = np.squeeze(SCfunc[:, :,ii] )  
    for i in range(len(xffin)): 
        curd = xffin[i,:] - np.mean(xffin[i,:])
        [dvxffin[ii,i], ccxffin[ii,i], Eps ] = stretching_current(ref = refd, cur = curd, dvmin = -Epsilon, dvmax = Epsilon, nbtrial = nbtrial,
                                                                  window = window_stretch, t_vec = t_vec )
                                                            

#%%  Compute dv/v from SC functions denoised with SVDWF (Pre-processing: removing the mean of each SC function before performing the stretching)   
dvwiener = np.zeros((2,int(len(SCfunc))))
ccwiener = np.zeros((2,int(len(SCfunc)))) # Set variables
errorwiener = np.zeros((2,int(len(SCfunc))))     
for ii in [0, 1]:
    if ii == 0:
        refd = np.squeeze(data['Wie_ref_NS']) 
        xfb_wiener = np.squeeze(data['SC_WIE_NS'] ) 
    elif ii == 1:
        refd = np.squeeze(data['Wie_ref_EW']) 
        xfb_wiener = np.squeeze(data['SC_WIE_EW'] )  
        
    for i in range(len(xfb_wiener)):
        curd = xfb_wiener[i] - np.mean(xfb_wiener[i])  
        [dvwiener[ii,i], ccwiener[ii,i], Eps ] = stretching_current(ref = refd, cur = curd, dvmin = -Epsilon, dvmax = Epsilon, nbtrial = nbtrial,
                                                                  window = window_stretch, t_vec = t_vec )
                                                              

#%%  Compute dv/v from SC functions denoised with ConvDeNoise (Pre-processing: removing the mean of each SC function before performing the stretching)   
dv = np.zeros((2,int(len(SCfunc))))
cc = np.zeros((2,int(len(SCfunc)))) 
error1 = np.zeros((2,int(len(SCfunc))))      
        
for ii in [0, 1]:
    if ii == 0:
        refd = np.squeeze(data['Conv_ref_NS'])
    elif ii == 1:
        refd = np.squeeze(data['Conv_ref_EW'])
    
    decoded_imgs_all_sq = np.squeeze(Denoised_SCF[:,:,ii])
    
    for i in range(len(decoded_imgs_all_sq)):
        curd = decoded_imgs_all_sq[i,:] - np.mean(decoded_imgs_all_sq[i,:])
        [dv[ii,i], cc[ii,i], Eps] = stretching_current(ref = refd, cur = curd, dvmin = -Epsilon, dvmax = Epsilon, nbtrial = nbtrial,
                                                                  window = window_stretch, t_vec = t_vec )
                                                                  
    
    
#%% Weighting average of the dv/v measurements over the components with the CC after stretching  following Hobiger et al. (2012) for the three methods
        
Sraw30 = []
Sdae30 = []
Swie30 = []
CCraw30 = []
CCdae30 = []
CCwie30 = []

for i in np.arange(len(dvxffin[0,:])):
    
    Sraw30.append((dvxffin[0,i]  * ccxffin[0,i]**2 + dvxffin[1,i]  * ccxffin[1,i]**2) / (ccxffin[0,i]**2 +ccxffin[1,i]**2))
    Sdae30.append((dv[0,i]  * cc[0,i]**2 + dv[1,i]  * cc[1,i]**2) / (cc[0,i]**2 +cc[1,i]**2))
    Swie30.append((dvwiener[0,i]  * ccwiener[0,i]**2 + dvwiener[1,i]  * ccwiener[1,i]**2) / (ccwiener[0,i]**2 +ccwiener[1,i]**2))
    
    CCraw30.append((ccxffin[0,i]**3 + ccxffin[1,i]**3) / (ccxffin[0,i]**2 +ccxffin[1,i]**2))
    CCdae30.append((cc[0,i]**3 + cc[1,i]**3) / (cc[0,i]**2 +cc[1,i]**2))
    CCwie30.append((ccwiener[0,i]**3 +   ccwiener[1,i]**3) / (ccwiener[0,i]**2 +ccwiener[1,i]**2))


#%% Plot to reproduce figures 7 of the paper
    
plt.rcParams.update({'font.size': 9}) 

##%% Set the x-axis for the data, do not change it!
datepl1 ='2017-09-25'
datepl2 =  '2017-10-11'
datozoom = pd.date_range(start=datepl1, end=datepl2,freq='0h20min')


if fig_choice == 7:
    xlio1 = datepl1 
    xlio2 = datepl2 
    ylio1 = -3
    ylio2 = 3
elif fig_choice == 8:
    xlio1 = '2017-09-26'
    xlio2 = '2017-10-01'
    ylio1 = -2
    ylio2 = 2


fig10 = plt.figure(figsize =(9, 9) )

plt.subplot(4,1,1)
plt.scatter(x= datozoom[:-1], y=Sraw30, s=15, c=np.abs(CCraw30), cmap='hot_r' )
plt.xlim(pd.Timestamp(xlio1), pd.Timestamp(xlio2))
plt.ylim(ylio1,ylio2)
plt.grid( linestyle='-', linewidth=.25)

plt.title('Raw',fontweight="bold")
plt.ylabel('$dv/v$ (%)')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0-.04, box.y0+.07, box.width , box.height+.02 ])

# Add rain patches
ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,9,27,23,10,0)), -3),
    .41, 15, alpha=.2))
ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,10,6,15,0,0)), -3),
    .67, 15, alpha=.2))

plt.clim(0, 1)
##%%

plt.subplot(4,1,2)
plt.scatter(x= datozoom[:-1], y=np.array(Sdae30), s=15, c=np.abs(CCdae30), cmap='hot_r')
plt.xlim(pd.Timestamp(xlio1), pd.Timestamp(xlio2))
plt.ylim(ylio1,ylio2)

plt.grid( linestyle='-', linewidth=.25)
plt.title('Denoised with ConvDeNoise',fontweight="bold")
plt.ylabel('$dv/v$ (%)')
ax = plt.gca()
ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,9,27,23,10,0)), -3),
    .41, 15, alpha=.2))

ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,10,6,15,0,0)), -3),
    .67, 15, alpha=.2))
plt.clim(0, 1)
box = ax.get_position()
ax.set_position([box.x0-.04, box.y0+.02, box.width  , box.height+.02 ])
cbaxes = fig10.add_axes([0.75, 0.675, 0.05, 0.02]) 
cb = plt.colorbar( cax = cbaxes, orientation = 'horizontal', ticks=[0 , .5 , 1])  #
cb.ax.set_title('CC after stretching' )

##%%

plt.subplot(4,1,3)
plt.scatter(x= datozoom[:-1], y=np.array(Swie30), s=15, c=np.abs(CCwie30), cmap='hot_r' )
plt.xlim(pd.Timestamp(xlio1), pd.Timestamp(xlio2))
plt.ylim(ylio1,ylio2)
plt.grid( linestyle='-', linewidth=.25)
plt.clim(0, 1)
plt.ylabel('$dv/v$ (%)')
plt.title('Denoised with SVDWF',fontweight="bold")
ax = plt.gca()
ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,9,27,23,10,0)), -3),
    .41, 15, alpha=.2))
ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,10,6,15,0,0)), -3),
    .67, 15, alpha=.2))
box = ax.get_position()
ax.set_position([box.x0-.04, box.y0-.03, box.width  , box.height+.02 ])


##%%
plt.subplot(4,1,4)
plt.plot(datozoom[:-1],np.squeeze(data['Precip']), 'dodgerblue',alpha = 1, linewidth =2)
plt.xlim(pd.Timestamp(xlio1), pd.Timestamp(xlio2))
plt.grid( linestyle='-', linewidth=.25)
plt.ylabel('Precipitation (mm/20-min)', color='dodgerblue')
plt.tick_params('y', colors='dodgerblue')


plt.xlabel('Date (yr-mth-day)')

ax = plt.gca()
ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,9,27,23,10,0)), -3),
    .41, 25, alpha=.2))
ax.add_patch(patches.Rectangle(
    (mdates.date2num(datetime.datetime(2017,10,6,15,0,0)), -3),
    .67, 25, alpha=.2))

ax4 = ax.twinx()

box = ax.get_position()
ax.set_position([box.x0-.04, box.y0-.05, box.width  , box.height ])
ax4.plot(datozoom[:-1]  , np.squeeze(data['Temp']), 'chocolate',alpha = 1, linewidth =2)
ax4.set_ylabel('Temperature ($^\circ$C)', color='chocolate')
ax4.tick_params('y', colors='chocolate')

plt.show()
