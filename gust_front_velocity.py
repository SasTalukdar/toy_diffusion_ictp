#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:48:37 2024

@author: sasankatalukdar
"""
import numpy as np
import matplotlib.pyplot as plt

def fun(t,r0,v0):
    return (3*(r0*r0*v0*t)**(1/3)+r0)/1000
    
ts=np.arange(0,3600,1)

plt.plot(ts,fun(ts,500,3),label='r0=500m, v0=3m/s')
plt.plot(ts,fun(ts,1000,3),label='r0=1000m, v0=3m/s')
plt.plot(ts,fun(ts,500,5),label='r0=500m, v0=5m/s')
plt.plot(ts,fun(ts,1000,5),label='r0=1000m, v0=5m/s')
plt.plot(ts,fun(ts,500,8),label='r0=500m, v0=8m/s')
plt.plot(ts,fun(ts,1000,8),label='r0=1000m, v0=8m/s')
plt.plot(ts,fun(ts,1000,1),label='r0=1000m, v0=1m/s')
plt.legend()
plt.ylabel('radius (km)')
plt.xlabel('time (s)')
plt.xlim(0,3600)
plt.ylim(-2,10)
plt.xticks(np.arange(0,3600,600))
plt.text(50,9,'$r = 3({r_o}^2.v_o.t)^{1/3}+r_o$')