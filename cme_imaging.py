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
from scipy.ndimage import gaussian_filter
from default_vals import default_cme_vals, default_star_vals, default_wind_vals


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


def stellar_radiance(wavelength, temp):
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
    spectral_intensity = (numerator/denominator).to(spectralIntensityUnits)
    return spectral_intensity.value



def calculate_stellar_luminosity(wavelength_range, temp, radius):
    wav_min = wavelength_range[0].to('nm').value
    wav_max = wavelength_range[1].to('nm').value
    d = check_units(temp=temp, radius=radius)
    radiance, toss = integrate.quad(stellar_radiance, wav_min, wav_max, (d['temp'].value))*un.erg/un.s/un.sr/un.cm**2
    lum = radiance * 4 * np.pi * un.sr * radius.to('cm')**2
    return lum


def calculate_stellar_radiance(wavelength_range, temp):
    wav_min = wavelength_range[0].to('nm').value
    wav_max = wavelength_range[1].to('nm').value
    d = check_units(temp=temp)
    radiance, toss = integrate.quad(stellar_radiance, wav_min, wav_max, (d['temp'].value))
    return radiance*un.erg/un.s/un.sr/un.cm**2


class Wind:
    def __init__(self, **kwargs):
        for k in default_wind_vals:
            if k in kwargs:
                self.__dict__.update({k:kwargs[k]})
            else:
                self.__dict__.update({k:default_wind_vals[k]})
                
        return_dict = check_units(vwind=self.vwind, Mdot=self.Mdot, linear_extent=self.linear_extent)
        if self.dim%2 == 0:
            self.dim += 1
            
        self.Mdot = return_dict['Mdot']
        self.vwind = return_dict['vwind']
        self.linear_extent = return_dict["linear_extent"]
        return
    
    
    def make_grids(self):
        x_grid = np.zeros((self.dim, self.dim))
        center = [int(self.dim/2+1), int(self.dim/2 +1)]
        for i in range(x_grid.shape[0]):
            x_grid[i][:] = np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        # not particles per se, but essentially the fraction of the pixel area that is covered by particle area?
        self.grid_scattering_factor = (const.sigma_T * self.Mdot/(8 * self.grid_distances * self.vwind * const.u.cgs)).to("")
        self.grid_scattering_factor[center,center] = np.nan
        return

###

class CME:
    def __init__(self, **kwargs):
        for k in default_cme_vals:
            if k in kwargs:
                self.__dict__.update({k:kwargs[k]})
            else:
                self.__dict__.update({k:default_cme_vals[k]})
                
        if self.dim%2 == 0:
            self.dim += 1
        
        self.grid_norm = None
        return
    
        
    def build_cme_normalized_grid(self, smear=False, smear_sig=3, **kwargs):
        for k in kwargs:
            self.__dict__.update({k:kwargs[k]})
            
        assert(self.dR_factor < 1)

        Rc = int(self.dim/2) +1
        
        dR = Rc * self.dR_factor
        grid = np.zeros((Rc,Rc))
        
        for j in range(int(dR/2)):
            Rmin = (Rc - dR*2) +j + dR*self.horizontal_height_factor
            Rmax = (Rc - dR) - j + dR*self.horizontal_height_factor
            
            phi_hw = np.pi/4
            a = np.pi/2 /phi_hw
            
            phi = np.linspace(0, np.pi/4, Rc)
            rmin = Rmin * np.cos(a * phi)
            rmax = Rmax * np.cos(a * phi)
            
            xmin, ymin = ((rmin*np.cos(phi)).astype(int), (self.vertical_height_factor*rmin*np.sin(phi)).astype(int))
            xmax, ymax = ((rmax*np.cos(phi)).astype(int), (self.vertical_height_factor*rmax*np.sin(phi)).astype(int))
                     
            xmin = xmin[::-1]
            ymin = ymin[::-1] 
            xmax = xmax[::-1]
            ymax = ymax[::-1] 
            
            grid_subset = np.zeros(grid.shape)
            for i in range(int(Rmax)):
                idxmax = np.where(xmax == i)[0]
                idxmin = np.where(xmin == i)[0]
            
                if len(idxmax) != 0:
                    idxmaxsave = idxmax[0]
                    yf_max = ymax[idxmax[0]]
                    if i > xmin.max():
                        yf_min = 0
                    if len(idxmin) != 0:
                        yf_min = ymin[idxmin[0]]
                        idxminsave = idxmin[0]
                        
                    if len(idxmin) == 0 and i < xmin.max():
                        yf_min = int((ymin[idxminsave] + ymin[idxminsave + 1])/2)
                
                if len(idxmax) == 0:
                    if len(idxmin) != 0:
                        yf_max = int((ymax[idxmaxsave] + ymax[idxmaxsave + 1])/2)
                        yf_min = ymin[idxmin[0]]
                        idxminsave = idxmin            
                        
                    if len(idxmin) == 0 and i < xmin.max():
                        yf_max = int((ymax[idxmaxsave] + ymax[idxmaxsave + 1])/2)
                        yf_min = int((ymin[idxminsave] + ymin[idxminsave + 1])/2)
                        
                    if len(idxmin) == 0 and i >= xmin.max():
                        yf_max = int((ymax[idxmaxsave] + ymax[idxmaxsave + 1])/2)
                        yf_min = 0
                try:
                    dy = np.abs(yf_max - yf_min)
                    grid_subset[i, yf_min:yf_max] = dy
                    
                    if yf_max == yf_min:
                        grid_subset[i, yf_max] = 1
                except:
                    pass
                    
            # grid += np.transpose(grid_subset) * ((self.depth_factor*(Rmin-Rmax)) * xmax )
            grid += np.transpose(grid_subset) * np.abs((Rmin-Rmax)  * xmax)** 0.5 * j 
                    
        grid_full = np.zeros((self.dim, self.dim))
        grid_full[Rc-1:,Rc-1:] = grid
        grid_full[0:Rc,Rc-1:] = np.flip(grid, axis = 0)
        if smear:
            grid_full = gaussian_filter(grid_full, sigma=smear_sig)
        self.grid_norm = grid_full/np.nansum(grid_full)
        return
    
        
    def make_grids(self, **kwargs):
        for k in kwargs:
            self.__dict__.update({k:kwargs[k]})
            
        if self.grid_norm is None:
            self.build_cme_normalized_grid()
            
        center = (self.grid_norm.shape[0]/2, self.grid_norm.shape[0]/2)

        x_grid = np.zeros(self.grid_norm.shape)
        center = [int(self.dim/2+1), int(self.dim/2 +1)]
        for i in range(x_grid.shape[0]):
            x_grid[i][:] = np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        self.grid_mass = self.grid_norm * self.mass
        self.grid_number = (self.grid_mass/const.u.cgs).to('')
        return 
    
    
    def convert_linear_to_angular(self, system_distance):
        self.grid_angular = (self.grid_distance/system_distance).to('')*un.rad.to('arcsec')
        self.system_distance = system_distance
        return self.grid_angular

###

class Star:
    def __init__(self, **kwargs):
        for k in default_star_vals:
            if k in kwargs:
                self.__dict__.update({k:kwargs[k]})
            else:
                self.__dict__.update({k:default_star_vals[k]})
                
        assert(len(self.wavelength_range) == 2), "wavelength_range should be list or numpy.ndarray of length 2"
        return_dict = check_units(temp=self.temp, radius=self.radius, distance=self.distance, linear_extent=self.linear_extent)
        self.temp = return_dict['temp']
        self.radius = return_dict['radius']
        self.luminosity = calculate_stellar_luminosity(self.wavelength_range, self.temp, self.radius)
        self.radiance = calculate_stellar_radiance(self.wavelength_range, self.temp)
        
        self.linear_extent = return_dict['linear_extent']
        self.distance = return_dict['distance']
        
        if self.dim % 2 == 0:
           self.dim += 1
           
        self.pix_res = self.linear_extent/self.dim
        self.center_wave = (self.wavelength_range[0] + self.wavelength_range[1])/2 
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
        x_grid = np.ones((self.dim, self.dim))
        center = [int(self.dim/2), int(self.dim/2)]
        x_grid = x_grid*np.abs(np.linspace(0, x_grid.shape[0]-1, x_grid.shape[0]) - center[0])            
        y_grid = x_grid.transpose()
        
        self.grid_distances = ((x_grid/self.dim)**2 + (y_grid/self.dim)**2)**0.5 * self.linear_extent
        
        thetamaxes = np.arctan((self.radius/np.sqrt(self.grid_distances**2 - self.radius**2)).to('')) 
        solid_angles = (np.pi/2*un.rad - thetamaxes)**2
        flux = (self.radiance * solid_angles * self.radius**2 / self.grid_distances**2).to('erg/s/cm**2')
        
        self.grid_stellar_energy_fluxes = flux
        self.grid_stellar_photon_fluxes = self.grid_stellar_energy_fluxes/self.photon_energy
        self.linear_to_angular()
        return
    
    
    def make_cme(self, smear=False, smear_sig=3, **kwargs):
        cme = CME(dim=self.dim)
        for k in kwargs:
            cme.__dict__.update({k:kwargs[k]})
            
        cme.build_cme_normalized_grid(smear=smear, smear_sig=smear_sig)
        cme.make_grids()
        self.cme = cme
        return
    
    
    def make_wind(self, **kwargs):
        wind = Wind()
        for k in kwargs:
            wind.__dict__.update({k:kwargs[k]})
        wind.dim = self.dim
        wind.make_grids()
        self.wind = wind
        return
    
    
    def particle_to_photon_flux(self):
        if self.cme is None:
            self.make_cme()
        if self.wind is None:
            self.make_wind()
            
        self.grid_wind_photons = self.wind.grid_scattering_factor * self.grid_stellar_photon_fluxes
        self.grid_cme_photons = (self.cme.grid_number * const.sigma_T * self.grid_stellar_photon_fluxes/self.pix_res**2).to("s**-1 * cm**-2")
        return 
    
    
    def linear_to_angular(self):
        dist_norm = self.grid_distances/(self.linear_extent)
        theta = (self.linear_extent/self.distance).to('') * un.rad.to('arcsec')
        self.grid_distances_angular = dist_norm*theta
        self.pix_res_ang = theta*un.arcsec/self.dim
        self.angular_extent = theta*un.arcsec
        return
    
    
    def plot(self, wind=True, cme=True, logscale=True, scale='linear'):
        
        dat = np.zeros(self.grid_cme_photons.shape) * (un.s * un.cm**2)**-1
        title = f"System distance : {self.distance}"
        if wind:
            dat += self.grid_scattering_factor
            title = f"{title}\nMdot={self.wind.Mdot}, v_wind={self.wind.vwind}"
        if cme:
            dat += self.grid_cme_photons
            title = f"{title}\n CME mass={self.cme.mass}"
        
        if scale.lower() == 'linear':
            # convert to flux received by a detector
            dat = (dat * self.pix_res**2/ (4 * np.pi * self.distance**2)).to('s**-1 * cm**-2')
            d = self.linear_extent.to('AU').value
            axis_label = 'Distance [AU]'
        elif scale.lower() == 'angular':
            d = self.angular_extent.to('arcsec').value
            axis_label = 'Distance [arcsec]'
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
        