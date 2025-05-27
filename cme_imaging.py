#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:55:05 2025

@author: idavis
"""
from ai.fri3d.model import StaticFRi3D
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un, constants as const
from scipy import integrate

class CME:
    def __init__(self, toroidal_height=1.5, half_width=45*np.pi/180, half_height=10*np.pi/180, pancaking=0.99, flattening=0.5, phi_dim: int = 256, theta_dim: int = 512, smooth=True, smooth_iter = 3, smooth_window=3, verbose=True):
        self.toroidal_height = toroidal_height
        self.half_width = half_width
        self.half_height = half_height
        self.pancaking = pancaking
        self.flattening = flattening 
        self.phi_dim = phi_dim
        self.theta_dim = theta_dim
        self.smooth_iter = smooth_iter
        self.smooth_window = smooth_window
        self.verbose = verbose
        self.wav_min = 100
        self.wav_max = 1000
        self.grid_norm = None
        self.linear_extent = None
        self.mass = None
        self.grid_distance = None
        self.grid_mass = None
        self.grid_number = None
        self.spectral_cube = None
        self.wavelengths = np.linspace(self.wav_min, self.wav_max, 10)
        self.sfr = None
        return
    
    def build_sfr(self, **kwargs):
        keys = ['toroidal_height', 'half_width', 'half_height', 'pancaking', 'flattening']
        kwarg_keys = list(kwargs.keys())
        for k in keys:
            if k not in kwarg_keys:
                print(f"{k} not in provided arguments, using value from the class instance")
                kwargs.update({f'{k}': self.__dict__[k]})
        sfr = StaticFRi3D(
            toroidal_height=kwargs["toroidal_height"],
            half_width=kwargs["half_width"],
            half_height=kwargs["half_height"],
            pancaking=kwargs["pancaking"],
            flattening=kwargs["flattening"],
        )
        self.sfr = sfr
                
        return
    
    def build_cme_normalized_grid(self, **kwargs):
        keys = ['toroidal_height', 'half_width', 'half_height', 'pancaking', 'flattening', 'phi_dim', 'theta_dim', 'smooth_iter', 'smooth_window', 'verbose']
        kwarg_keys = list(kwargs.keys())
        
        for k in keys:
            if k not in kwarg_keys:
                print(f"{k} not in provided arguments, using value from the class instance")
                kwargs.update({f'{k}': self.__dict__[k]})
                
        if kwargs['verbose']:
            print("Building CME model")
            
        sfr = StaticFRi3D(
            toroidal_height=kwargs["toroidal_height"],
            half_width=kwargs["half_width"],
            half_height=kwargs["half_height"],
            pancaking=kwargs["pancaking"],
            flattening=kwargs["flattening"],
        )
        self.sfr = sfr
        
        phi = np.linspace(-sfr.half_width, sfr.half_width, kwargs["phi_dim"])
        x, y, z = sfr.shell(phi = phi, theta=np.linspace(0, np.pi * 2, kwargs["theta_dim"]))
        
        xt = x.transpose() * (x.shape[1] - 1)/x.max()
        y_trans = y + y.max()
        yt = y_trans.transpose() * (y_trans.shape[1] - 1)/y_trans.max()
        zt = z.transpose()
        
        grid = np.zeros((xt.shape[0], yt.shape[0]))
        if kwargs["verbose"]:
            print("Flattening CME along z axis")
            
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                idxs = np.where((i <= xt) & (xt < i+1) & (j <= yt) & (yt < j+1))
                vals = np.abs(np.nansum(zt[idxs])*(xt[idxs]))
                if vals.size != 0:
                    grid[j,i] = np.mean(vals)
        
        if kwargs["verbose"]:
            print("Filling gaps in flattened grid")
        
        nsmooth = kwargs["smooth_window"]
        zero_idxs = np.where(grid == 0)
        for idx in range(len(zero_idxs[0])):
            x0 = zero_idxs[0][idx]-nsmooth
            xe = zero_idxs[0][idx]+nsmooth
            y0 = zero_idxs[1][idx]-nsmooth
            ye = zero_idxs[1][idx]+nsmooth
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            dat_rang = grid[x0:xe, y0:ye]
            grid[zero_idxs[0][idx], zero_idxs[1][idx]] = np.nanmedian(dat_rang)
            
        if kwargs["verbose"]:
            print("Transferring flattened model to larger grid and smoothing")
        
        grid_copy = np.zeros((grid.shape[0]*2, grid.shape[1]*2))
        dx = int(grid.shape[0]/2)
        dy = int(grid.shape[1]/2)
        grid_copy[grid.shape[1]-dy: grid.shape[1]+dy, 0:int(dx*2)] = grid
        niter = kwargs["smooth_iter"]
        iter_count = 1
        
        while iter_count <= niter:
            for i in range(grid_copy.shape[0]):
                x0 = i - nsmooth
                xe = i + nsmooth
                if x0 < 0:
                    x0 = 0
                for j in range(grid_copy.shape[1]):
                    y0 = j - nsmooth
                    ye = j + nsmooth
                    if y0 < 0:
                        y0 = 0
                    dat_rang = grid_copy[y0:ye, x0:xe]
                    grid_copy[j,i] = np.nanmean(dat_rang)
            iter_count += 1
            
        grid_norm = grid_copy/np.sum(grid_copy)
        self.grid_norm = grid_norm
        return grid_norm

    def make_grids(self, mass = 1e19*un.g, linear_extent= 6*un.AU):
        if self.grid_norm is None:
            self.build_cme_normalized_grid()
            
        center = (0, self.grid_norm.shape[0]/2)
        d = linear_extent
        x_grid = np.zeros(self.grid_norm.shape)
        for i in range(x_grid.shape[0]):
            x_grid[i][:] = np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0]
            
        y_grid = np.zeros(self.grid_norm.shape)
        for i in range(y_grid.shape[1]):
            y_grid[:][i] = np.abs(np.linspace(0, y_grid.shape[1]-1, y_grid.shape[1]) - center[1])
        y_grid = y_grid.transpose()
        
        self.linear_extent = linear_extent
        self.mass = mass
        self.grid_distance = (x_grid**2 + y_grid**2)/(self.grid_norm.shape[0] * self.grid_norm.shape[1])*d.to('cm')
        self.grid_mass = self.grid_norm * mass
        self.grid_number = (self.grid_mass/const.m_p).to('')
        return 

    def make_spectral_grid(self, temp, radius, wav_min, wav_max):
        if type(radius) != un.Quantity:
            print("assuming radius is in units of solar radii")
            radius = radius*const.R_sun
        if type(temp) != un.Quantity:
            print("assuming temperature is in Kelvin")
            temp *= un.K
        if type(wav_min) != un.Quantity:
            print("assuming wavelength is in nanometers")
            wav_min *= un.nm
        if type(wav_max) != un.Quantity:
            print("assuming wavelength is in nanometers")
            wav_max *= un.nm
        print(f"{wav_min}, {wav_max}")
        # print(f"{stellar_spectral_luminosity(temp, radius, wav_min)}, {stellar_spectral_luminosity(temp, radius, wav_max)}")
        intensity, toss = integrate.quad(stellar_spectral_luminosity, wav_min.to('nm').value, wav_max.to('nm').value, (temp.value, radius.to('R_sun').value))
        flux_grid = intensity/ (4 * np.pi * self.grid_distance**2) * un.erg/un.s * self.grid_number * const.sigma_T
        return flux_grid, intensity * un.erg/un.s

    def make_spectral_cube(self, temp, radius, wav_min=None, wav_max=None, n_grids=10):
        if self.grid_mass is None:
            self.make_grids()
            
        if wav_min is None:
            wav_min = self.wav_min
        elif wav_min is not None:
            self.wav_min = wav_min
        if wav_max is None:
            wav_max = self.wav_max
        elif wav_max is not None:
            self.wav_max = wav_max
            
        if type(radius) != un.Quantity:
            print("assuming radius is in units of solar radii")
            radius = radius*const.R_sun
        if type(temp) != un.Quantity:
            print("assuming temperature is in Kelvin")
            temp *= un.K
        if type(wav_min) != un.Quantity:
            print("assuming wavelength is in nanometers")
            wav_min *= un.nm
        if type(wav_max) != un.Quantity:
            print("assuming wavelength is in nanometers")
            wav_max *= un.nm
        
        spectral_cube = np.zeros((n_grids, self.grid_number.shape[0], self.grid_number.shape[1]))
        wavelengths = np.linspace(wav_min.value, wav_max.value, n_grids+1)
        for i in range(n_grids):
            cme, star = self.make_spectral_grid(temp, radius, wavelengths[i], wavelengths[i+1])
            spectral_cube[i] = cme
        self.spectral_cube = spectral_cube*un.erg/un.s
        self.wavelengths = np.linspace(wav_min.value, wav_max.value, n_grids)
        return 
    
    def plot_cme3D(self):
        phi = np.linspace(-self.sfr.half_width, self.sfr.half_width, self.phi_dim)
        x, y, z = self.sfr.shell(phi = phi, theta=np.linspace(0, np.pi * 2, self.theta_dim))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", adjustable="box")
        ax.plot_wireframe(x, y, z, alpha=0.4)
        ax.set_xlim3d(0.0, 3)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        return
    
    def plot_cme_spectrum(self):
        vals = np.zeros(self.spectral_cube.shape[0])
        for i, grid in enumerate(self.spectral_cube):
            grid[grid == np.inf] = 0
            vals[i] = np.nansum(grid)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.wavelengths, vals)
        return fig, ax
    
    def convert_energy_to_photon_flux(self):
        return
    
    def convert_linear_to_angular(self, system_distance):
        return
    
    
    def save(self, fn):
        return
    
    def load_cme(self, fn):
        dat = np.load(fn)
        return


def stellar_spectral_luminosity(wavelength, temp, radius):
    if type(radius) != un.Quantity:
        # print("assuming radius is in units of solar radii")
        radius = radius*const.R_sun
    if type(temp) != un.Quantity:
        # print("assuming temperature is in Kelvin")
        temp *= un.K
    if type(wavelength) != un.Quantity:
        # print("assuming wavelength is in nanometers")
        wavelength *= un.nm
        
    spectralIntensityUnits = 'erg*s**-1*cm**-2*nm**-1'
    h = const.h
    c = const.c
    k = const.k_B
    numerator = 2 * h*c**2/wavelength**5
    denominator = np.exp((h*c/(wavelength*k*temp)).to('')) -1
    spectralIntensity = (numerator/denominator).to(spectralIntensityUnits)
    spectral_luminosity = (spectralIntensity*4*np.pi*radius**2).to('erg/s/nm')
    return spectral_luminosity.value