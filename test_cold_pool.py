#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:18:36 2024

@author: sasankatalukdar
"""

import xarray as xr
import numpy as np

file=xr.open_dataset('td_maps_diffK20000.0_tausub15.0_crhad15.0_cin_radius10_diurn0.nc')
cin=file['CIN'][10,:,:].values
cin9=file['CIN'][9,:,:].values
cin_grad=np.sqrt(np.nansum(np.array(np.gradient(cin))**2,axis=0))
cin_grad9=np.sqrt(np.nansum(np.array(np.gradient(cin9))**2,axis=0))

from matplotlib import pyplot as plt
plt.imshow(cin)


import numpy as np
from scipy.ndimage import convolve

def find_0_nearby_1(x):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors_are_1 = convolve((x == 1).astype(int), kernel, mode='constant', cval=0) > 0
    x[(x == 0) & neighbors_are_1] = 2
    x[x==1]=0
    x[x==2]=1
    return x

def find_max_grad(cin_grad,cnv_idx):
    cp_locs=np.zeros(np.shape(cin_grad))
    for i0,j0 in cnv_idx:
        found_xr=0
        found_xl=0
        found_y=0
        x=0
        while found_xr==0 or found_xl==0:
            y=0
            while found_y==0:
                if min(j0+y-1,i0+x-1)<=0 or max(j0+y+1,i0+x+1)>=np.shape(cp_locs)[0]:
                    break
                if cin_grad[i0+x,j0+y]>cin_grad[i0+x,j0+y+1] and cin_grad[i0+x,j0+y]>=cin_grad[i0+x,j0+y-1]:
                    cp_locs[i0+x,j0+y]=1
                    found_y=1
                    y=0
                else:
                    y+=1
                
            found_y=0
            while found_y==0:
                if min(j0+y-1,i0+x-1)<=0 or max(j0+y+1,i0+x+1)>=np.shape(cp_locs)[0]:
                    break
                if cin_grad[i0+x,j0+y]>cin_grad[i0+x,j0+y+1] and cin_grad[i0+x,j0+y]>=cin_grad[i0+x,j0+y-1]:
                    cp_locs[i0+x,j0+y]=1
                    found_y=1
                    y=0
                else:
                    y-=1
                
            if found_xr==0:
                if min(j0+y-1,i0+x-1)<=0 or max(j0+y+1,i0+x+1)>=np.shape(cp_locs)[0]:
                    found_xr=1
                    pass
                elif cin_grad[i0+x,j0+y]>cin_grad[i0+x-1,j0] and cin_grad[i0+x,j0]>=cin_grad[i0+x+1,j0]:
                    cp_locs[i0+x,j0]=1
                    found_xr=1
                    x=0
                else:
                    x+=1
            if found_xl==0 and found_xr==1:
                if min(j0+y-1,i0+x-1)<=0 or max(j0+y+1,i0+x+1)>=np.shape(cp_locs)[0]:
                    found_xl=1
                    pass
                elif cin_grad[i0+x,j0+y]>cin_grad[i0+x-1,j0] and cin_grad[i0+x,j0]>=cin_grad[i0+x+1,j0]:
                    cp_locs[i0+x,j0]=1
                    found_xl=1
                else:
                    x-=1
    return cp_locs


            
# add another mask using a gradient threshold
                    
        
        
    