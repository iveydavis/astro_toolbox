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



def check_units(**kwargs):
    kwarg_keys = list(kwargs.keys())
    return_dict = {}
    for k in kwarg_keys:
        if 'radius' in k:
            radius = kwargs[k]
            if type(radius) != un.Quantity:
                print("assuming radius is in units of solar radii")
                radius = radius*const.R_sun
            return_dict.update({k:radius.to('R_sun')})
            
        if 'temp' in k:
            temp = kwargs[k]
            if type(temp) != un.Quantity:
                print("assuming temperature is in units of Kelvin")
                temp = temp*un.K
            return_dict.update({k:temp.to('K')})
            
        if 'linear_extent' in k:
            linear_extent = kwargs[k]
            if type(linear_extent) != un.Quantity:
                print("assuming linear extent in units of AU")
                linear_extent *= un.AU
            return_dict.update({k:linear_extent.to("AU")})
                
        if "distance" in k or "dist" in k:
            distance = kwargs[k]
            if type(distance) != un.Quantity:
                print("assuming system distance is in units of parsec")    
                distance = distance*un.pc
            return_dict.update({k:distance.to('pc')})
                
        if "Mdot" in k or "mdot" in k:
            Mdot = kwargs[k]
            if type(Mdot) != un.Quantity:
                print("assuming Mdot is in units of solar masses per year")
                Mdot = Mdot * un.M_sun/un.yr
            return_dict.update({k:Mdot.to('M_sun/yr')})
            
        if "vwind" in k:
            vwind = kwargs[k]
            if type(vwind) != un.Quantity:
                print("assuming wind is in km/s")
                vwind = vwind * un.km/un.s
            return_dict.update({k:vwind.to('km/s')})
    return return_dict


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

def calculate_stellar_luminosity(wavelength_range, temp, radius):
    wav_min = wavelength_range[0].to('nm').value
    wav_max = wavelength_range[1].to('nm').value
    d = check_units(temp=temp, radius=radius)
    lum, toss = integrate.quad(stellar_spectral_luminosity, wav_min, wav_max, (d['temp'].value, d['radius'].value))
    return lum*un.erg/un.s


class Wind:
    def __init__(self, dim:int, Mdot, vwind, linear_extent, wavelength_range=[100,1000]*un.nm):
        if dim % 2 == 0:
           dim += 1 
        return_dict = check_units(vwind=vwind, Mdot=Mdot, linear_extent=linear_extent)
        assert(len(wavelength_range) == 2), "wavelength_range should be list or array of length 2"
        
        self.dim = dim
        self.wavelength_range = wavelength_range
        self.center_lambda = (wavelength_range[0] + wavelength_range[1])/2
        
        self.Mdot = return_dict['Mdot']
        self.vwind = return_dict['vwind']
        self.linear_extent = return_dict["linear_extent"]
        self.make_grids()
        return
    
    def make_grids(self):
        x_grid = np.zeros((self.dim, self.dim))
        center = [int(self.dim/2+1), int(self.dim/2 +1)]
        for i in range(x_grid.shape[0]):
            x_grid[i][:] = np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        # not particles per se, but essentially the fraction of the pixel area that is covered by particle area?
        self.grid_wind_particles = (const.sigma_T * self.Mdot/(8 * self.grid_distances * self.vwind * const.u.cgs)).to("")
        return


class CME:
    def __init__(self, toroidal_height=1.5, half_width=45*np.pi/180, half_height=10*np.pi/180, pancaking=0.99, flattening=0.5, phi_dim: int = 256, theta_dim: int = 512, smooth=True, smooth_iter = 3, smooth_window=3, dim=1024, verbose=True):
        self.verbose = verbose
        
        # cme model definitions:
        self.toroidal_height = toroidal_height
        self.half_width = half_width
        self.half_height = half_height
        self.pancaking = pancaking
        self.flattening = flattening 
        self.phi_dim = phi_dim
        self.theta_dim = theta_dim
        self.sfr = None
        
        # smoothing information for after collapse along z axis:
        self.smooth_iter = smooth_iter
        self.smooth_window = smooth_window
        
        self.linear_extent = None
        self.mass = None
        
        if dim % 2 == 0:
           dim += 1 
           
        self.dim = dim
        self.grid_norm = None
        self.grid_distances = None
        self.grid_mass = None
        self.grid_number = None
        return
    
    def build_sfr(self, verbose=False, **kwargs):
        keys = ['toroidal_height', 'half_width', 'half_height', 'pancaking', 'flattening']
        kwarg_keys = list(kwargs.keys())
        for k in keys:
            if k not in kwarg_keys:
                if verbose:
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
    
    def build_cme_normalized_grid(self, force_symmetry=True, halo=False, **kwargs):
        keys = ['toroidal_height', 'half_width', 'half_height', 'pancaking', 'flattening', 'phi_dim', 'theta_dim', 'smooth_iter', 'smooth_window', 'verbose']
        kwarg_keys = list(kwargs.keys())
        
        for k in keys:
            if k not in kwarg_keys:
                print(f"{k} not in provided arguments, using value from the class instance")
                kwargs.update({f'{k}': self.__dict__[k]})
                
        if kwargs['verbose']:
            print("Building CME model")
            
        if self.sfr is None and 'sfr' not in kwarg_keys:
            self.build_sfr(verbose=False)
            sfr = self.sfr
        
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
                # vals = np.abs(np.nansum(zt[idxs])*(xt[idxs]))
                vals = np.abs(zt[idxs[0]] - zt[idxs[-1]])
                if vals.size != 0:
                    grid[j,i] = np.mean(vals)
        
        if kwargs["verbose"]:
            print("Filling gaps in flattened grid")
        
        nsmooth = kwargs["smooth_window"]
        zero_idxs = np.where(grid == 0)
        smooth_count = 1
        count = 1
        while count <= smooth_count:
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
            zero_idxs = np.where(grid == 0)
            count +=1  
        
        
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
        
        siz = int(grid_copy.shape[0]/2)    
        if force_symmetry:    
            bottom_half = grid_copy[:siz,:]
            grid_copy[:siz, :] = bottom_half
            grid_copy[siz:, :] = np.flip(bottom_half, axis=0)
            
        
            
        final_grid = np.zeros((self.dim, self.dim))
        center_final = (int(self.dim/2 + 1), int(self.dim/2 + 1))
        final_grid[center_final[0]-siz:center_final[0]+siz, center_final[0]:int(center_final[0]+siz*2)] = grid_copy
        
        if halo:
            new_grid = np.zeros(final_grid.shape)
            center = int(final_grid.shape[0]/2)-1
            gridv = final_grid[center,center:]
            theta_vec = np.linspace(0*un.rad, np.pi*2*un.rad, int(self.dim)*4)
            for i, v in enumerate(gridv):
                x = i * np.cos(theta_vec)
                y = i * np.sin(theta_vec)
                x = np.round(x).astype(int)
                y = np.round(y).astype(int)
                new_grid[y+center, x+center] = v
            final_grid = new_grid  
        
        grid_norm = final_grid/np.sum(final_grid)
        self.grid_norm = grid_norm
        return 
    
    def make_grids(self, mass = 1e19*un.g, linear_extent= 6*un.AU):
        if self.grid_norm is None:
            self.build_cme_normalized_grid()
            
        center = (self.grid_norm.shape[0]/2, self.grid_norm.shape[0]/2)
        
        if linear_extent is None:
            linear_extent = self.linear_extent
        elif linear_extent is not None:
            self.linear_extent = linear_extent
        
        if mass is None:
            mass = self.mass
        elif mass is not None:
            self.mass = mass
        
        x_grid = np.zeros((self.dim, self.dim))
        center = [int(self.dim/2+1), int(self.dim/2 +1)]
        for i in range(x_grid.shape[0]):
            x_grid[i][:] = np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        self.grid_mass = self.grid_norm * mass
        self.grid_number = (self.grid_mass/const.u.cgs).to('')
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
    
    def convert_linear_to_angular(self, system_distance):
        self.grid_angular = (self.grid_distance/system_distance).to('')*un.rad.to('arcsec')
        self.system_distance = system_distance
        return self.grid_angular


class Star:
    def __init__(self, temp, radius, distance, dim, linear_extent, wavelength_range=[100*un.nm,1000*un.nm]):
        assert(len(wavelength_range) == 2), "wavelength_range should be list or numpy.ndarray of length 2"
        return_dict = check_units(temp=temp, radius=radius, distance=distance, linear_extent=linear_extent)
        self.temp = return_dict['temp']
        self.radius = return_dict['radius']
        self.luminosity = calculate_stellar_luminosity(wavelength_range, self.temp, self.radius)
        self.linear_extent = return_dict['linear_extent']
        self.distance = return_dict['distance']
        
        if dim % 2 == 0:
           dim += 1 
        self.dim = dim
        self.pix_res = self.linear_extent/self.dim
        self.wavelength_range = wavelength_range
        self.center_wave = (wavelength_range[0] + wavelength_range[1])/2 
        self.photon_energy = (const.h*const.c/self.center_wave).to('erg')
        self.make_stellar_grids()
        
        self.cme = None
        self.wind = None
        return
    
    def reset_spectral_properties(self, wavelength_range):
        self.wavelength_range = wavelength_range
        self.center_wave = (wavelength_range[0] + wavelength_range[1])/2 
        self.photon_energy = (const.h*const.c/self.center_wave).to('erg')
        self.make_stellar_grids()
        return
    
    def make_stellar_grids(self):
        x_grid = np.zeros((self.dim, self.dim))
        center = [int(self.dim/2+1), int(self.dim/2 +1)]
        for i in range(x_grid.shape[0]):
            x_grid[i][:] = np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        self.grid_stellar_energy_fluxes = (self.luminosity/(4 * np.pi *self.grid_distances**2)).to('erg/s/cm**2')
        self.grid_stellar_photon_fluxes = self.grid_stellar_energy_fluxes/self.photon_energy
        self.linear_to_angular()
        return
    
    def make_cme(self, mass=1e19*un.g, toroidal_height=1.5, half_width=45*np.pi/180, half_height=10*np.pi/180, pancaking=0.99, flattening=0.5, smooth=True, smooth_iter = 3, smooth_window=3, verbose=True, halo=False):
        cme = CME(toroidal_height=toroidal_height, half_width=half_width, half_height=half_height, pancaking=pancaking, flattening=flattening, phi_dim=int((self.dim-1)/8), theta_dim=int((self.dim-1)/4), smooth=smooth, smooth_iter =smooth_iter, smooth_window=smooth_window, dim=self.dim, verbose=True)
        cme.build_cme_normalized_grid(halo=halo)
        cme.make_grids(mass=mass, linear_extent=self.linear_extent)
        self.cme = cme
        return
    
    
    def make_wind(self, Mdot=30 * 2e-14 * un.M_sun/un.yr, vwind=400*un.km/un.s):
        wind = Wind(self.dim, Mdot=Mdot, vwind=vwind, linear_extent=self.linear_extent)
        self.wind = wind
        return
    
    def particle_to_photon_flux(self):
        if self.cme is None:
            self.make_cme()
        if self.wind is None:
            self.make_wind()
            
        self.grid_wind_photons = self.wind.grid_wind_particles * self.grid_stellar_photon_fluxes
        self.grid_cme_photons = (self.cme.grid_number * const.sigma_T * self.grid_stellar_photon_fluxes/self.pix_res**2).to("s**-1 * cm**-2")
        return 
    
    def linear_to_angular(self):
        dist_norm = self.grid_distances/(self.linear_extent)
        theta = (self.linear_extent/self.distance).to('') * un.rad.to('arcsec')
        self.grid_distances_angular = dist_norm*theta
        self.pix_res_ang = theta*un.arcsec/self.dim
        self.angular_extent = theta*un.arcsec
        return
    
    def plot(self, wind=True, cme=True, logscale=True):
        dat = np.zeros(self.grid_cme_photons.shape) * (un.s * un.cm**2)**-1
        title = f"System distance : {self.distance}"
        if wind:
            dat += self.grid_wind_photons
            title = f"{title}\nMdot={self.wind.Mdot}, v_wind={self.wind.vwind}"
        if cme:
            dat += self.grid_cme_photons
            title = f"{title}\n CME mass={self.cme.mass}"
        # convert to flux received by a detector
        dat = (dat * self.pix_res**2/ (4 * np.pi * self.distance**2)).to('s**-1 * cm**-2')
        
        d = self.linear_extent.to('AU').value
        axis_label = 'Distance [AU]'
        cbar_label = r'photons/s/cm$^2$'
            
        if logscale:
            dat = np.log10(dat.value)
            cbar_label = f"log10({cbar_label})"
            
        extents = [-d/2, d/2,-d/2, d/2]
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(dat, extent=extents)

        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        ax.set_title(title)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)
        