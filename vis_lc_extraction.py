import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.ndimage.filters import uniform_filter

from astropy.io import fits
from astropy import constants as const, units as un
from astropy.time import Time
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from photutils.aperture import CircularAperture, aperture_photometry
from photutils import DAOStarFinder, EllipticalAperture
from SchedulingTools import AccessSIMBAD, ApplyProperMotion

import matplotlib.pyplot as plt
import palettable
import matplotlib.colors as mc
from matplotlib import patches
import matplotlib as mp
import colorsys
import aplpy

import os
import bdsf
import sys
import glob
from PIL import Image
import warnings
import copy


def adjust_lightness(color, amount = 0.5):
    """
    Adjusts lightness/darkness of a color (not changing opacity)
    :param color: color to adjust
    :type color: str
    :param amount: amount to adjust color; higher is brighter and lower is darker
    :type amount: float
    :return: The adjusted color
    :rtype: colorsys
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1,amount*c[1])), c[2])


def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Function for making a 2D Gaussian, raveled to 1D array
    :param xy: The 2D x-y array that the Gaussian is constructed in
    :type xy: np.ndarray
    :param amplitude: amplitude of the Gaussian peak
    :type amplitude: float
    :param xo: offset from 0 of gaussian in x direction
    :type xo: float
    :param yo: offset from 0 of gaussian in y direction
    :type yo: float
    :param sigma_x: standard deviation (width of Gaussian) in x-direction
    :type sigma_x: float
    :param sigma_y: standard deviation (width of Gaussian) in y-direction
    :type sigma_y: float
    :param theta: rotation of gaussian axis
    :type theta: float, radians
    :param offset: overall offset of Guassian intensity from 0. Should be 0 for making a psf
    :type offset: float
    :return: the 2D gaussian
    :rtype: np.array
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def theta_to_bpa(theta):
    """
    Converts beam position angle (BPA) to degrees
    :param theta: BPA, generally taken from FITS file header
    :type theta: float
    :rtype: float; deg
    """
    n_rot = int(theta/2/np.pi) # integer number of 2pi that the angle has been offset
    theta -= n_rot * 2 * np.pi
    theta *= -1
    bpa = theta*un.rad.to('deg')
    bpa -= 90
    return bpa


def get_beam_shape_params_circ(fn, dpix=12, update_file_header=True, threshold=8, fwhm=8):
    """
    Gets the beam shape parameters that can be used in twoD_Gaussian to construct a psf. Necessary if the data were not deconvolved. Assumes that the sources are sufficiently circular that DAOStarFinder can identify to extract-- if this doesn't work, consider using get_beam_shape_params_iter
    :param fn: the filename of the image to be used to derive beam shape parameters
    :type fn: str
    :param dpix: defines the shape of the array for Gaussian construction; creates array of dimension 2 x (2 * dpix)
    :type dpix: int
    :param update_file_header: if True, updates the BPA, BMIN, and BMAJ values in the header of the file
    :type update_file_header: bool
    :param threshold: the source-detection threshold. Uses the brightest source to derive the beam shape
    :type threshold: float
    :param fwhm: a preliminary fwhm (pixels) to identify sources
    :type fwhm: float
    :return: the solutions to the 2D gaussian equation 
    :rtype: np.array len 7
    """     
    f = fits.open(fn)[0]
    dat = f.data[0,0,:,:]
    
    thresh = np.median(dat) + threshold*np.std(dat)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh)
    sources = daofind(dat)
    if sources is not None:
        sources.sort('flux', reverse=True) # make sure the brightest source is listed first
        y,x = sources[0]['xcentroid','ycentroid'] # get the coordinates of the source
        xarr = np.linspace(0, 2 * dpix-1, 2 * dpix)
        yarr = np.linspace(0, 2 * dpix-1, 2 * dpix)

        xarr, yarr = np.meshgrid(xarr, yarr) # define the grid for the gaussian solution to be sought
        isolated_source = dat[int(x-dpix):int(x+dpix),int(y-dpix):int(y+dpix)]
        p0 = (sources[0]['peak'], dpix, dpix, 3, 3, 0 , np.nanmedian(dat))
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, (xarr, yarr), isolated_source.ravel(), p0=p0 )
        except:
            # in case the gaussian solution fails
            warnings.warn(f"Could not get beam shape for {fn}")
            popt = np.array([0,0,0,0,0,0,0])
            
    elif sources is None:
        # if there were never any sources identified to begin with:
        popt = np.array([0,0,0,0,0,0,0])
        
    if update_file_header:
        # save solutions to the header of the file:
        hdr = f.header
        hdr['BMAJ'] = np.abs(popt[3] * hdr['CDELT2'] * 3.5) # CDELT2 is the pixel size; this recasts from pixels to degrees. factor of 3.5 to account for beam underestimation
        hdr['BMIN'] = np.abs(popt[4] * hdr['CDELT2'] * 3.5)
        hdr['BPA'] = theta_to_bpa(popt[5])

        f.header = hdr
        f.writeto(fn, overwrite=True)

    return popt


def get_beam_shape_params_iter(fn, update_file_header=True, subframe_radius = (50,50), std_threshold = 6, niter = 21, sma=15, eps=0.7, pa=2):
    """
    Gets the beam shape parameters that can be used in twoD_Gaussian to construct a psf. Necessary if the data were not deconvolved. This is slower than get_beam_shape_params_circ, but can identify even extremely elliptical sources.
    :param fn: the filename of the image to be used to derive beam shape parameters
    :type fn: str
    :param update_file_header: if True, updates the BPA, BMIN, and BMAJ values in the header of the file
    :type update_file_header: bool
    :param subframe_radius: defines the shape of the array for a subframe around the source
    :type dpix: tuple len 2
    :param std_threshold: the threshold for making sure there are not other sources in the subframe-- the higher std_threshold, the higher the likelihood there's another source
    :type std_threshold: float
    :param niter: the number of iterations to go through to try to identify a source
    :type niter: int
    :param fwhm: a preliminary fwhm (pixels) to identify sourcs
    :type fwhm: float
    :return: the solutions to the 2d gaussian equation
    :rtype: np.array len 7
    """   
    
    f = fits.open(fn)[0]
    data = f.data
    data_new = copy.deepcopy(data)
    thresh = 5*std_threshold # preliminary threshold to enter the while loop
    iter_count = 1 # keeps track of the iteration
    data_std = np.nanstd(data)
    while thresh > std_threshold and iter_count < niter:
        idx_max = np.where(data_new == np.nanmax(data_new))
        x_max, y_max= int(idx_max[0][0]), int(idx_max[1][0])
        subframe = data_new[x_max-subframe_radius[0]:x_max+subframe_radius[0], y_max-subframe_radius[1]:y_max+subframe_radius[1]]
        thresh = np.nanstd(subframe)/data_std
        if thresh >= std_threshold:
            data_new[x_max-subframe_radius[0]:x_max+subframe_radius[0], y_max-subframe_radius[1]:y_max+subframe_radius[1]] = np.nan
        iter_count += 1
    
    if thresh >= std_threshold:
        # if it never reaches a std that suggests there's only one source in the subframe to extract a solution from
        warnings.warn(f"STD threshold was never met ({thresh})")
        popt = np.array([0,0,0,0,0,0,0])
    
    try:
        xarr = np.linspace(0, subframe.shape[0]-1, subframe.shape[0])
        yarr = np.linspace(0, subframe.shape[1]-1, subframe.shape[1])
        xarr, yarr = np.meshgrid(xarr, yarr)
        p0 = (np.nanmax(subframe), subframe.shape[0]/2, subframe.shape[1]/2, 3, 3, 0 , np.nanmedian(subframe))
        popt, pcov = opt.curve_fit(twoD_Gaussian, (xarr, yarr), subframe.ravel(), p0=p0)
        
    except:
        warnings.warn("Could not get beam parameters")
        popt = np.array([0,0,0,0,0,0,0])
        
    if update_file_header:
        # save solutions to the header of the file:
        hdr = f.header
        hdr['BMAJ'] = np.abs(popt[3] * hdr['CDELT2'] * 3.5) # CDELT2 is the pixel size; this recasts from pixels to degrees. factor of 3.5 to account for beam underestimation
        hdr['BMIN'] = np.abs(popt[4] * hdr['CDELT2'] * 3.5)
        hdr['BPA'] = theta_to_bpa(popt[5])

        f.header = hdr
        f.writeto(fn, overwrite=True)
        
    return popt


def rms(data):
    rms = np.sqrt(np.nanmean(data**2))
    return rms

#%%
class beam:
    def __init__(self, bmaj:"pix", bmin:"pix", bpa:"deg"):
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        return
    

class radio_source:
    def __init__(self, label:str, current_position:SkyCoord, persistent=None, seps=None, pos_angs=None, avg_position=None, frames=None):
        self.label = label
        self.persistent = persistent
        self.seps = seps
        self.pos_angs = pos_angs
        self.avg_position = avg_position
        self.current_position = current_position
        self.frames = frames
        return
    
    def update_frames(self, frame_idx:int, concat=True):
        if self.frames is not None:
            if concat:
                self.frames = np.concatenate((self.frames, [frame_idx]))
        if self.frames is None or not concat:
            self.frames = np.array([frame_idx])
        return
    
    def update_avg_position(self, avg_position:SkyCoord):
        self.avg_position = avg_position
        return
    
    def update_persistent(self, persistent:bool):
        self.persistent = persistent
        return
    
    def update_seps(self, sep, concat=True):
        if self.seps is not None:
            if concat:
                self.seps = np.concatenate((self.seps, [sep]))
        if self.seps is None or not concat:
            self.seps = np.array([sep])
        return
    
    def update_pos_angs(self, pos_ang, concat=True):
        if self.pos_angs is not None:
            if concat:
                self.pos_angs = np.concatenate((self.pos_angs, [pos_ang]))
        if self.pos_angs is None or not concat:
            self.pos_angs = np.array([pos_ang])
        return
    
    def cross_match(self, source_list, max_sep):
        source_pos = SkyCoord([s.current_position for s in source_list])
        cm_idx, cm_sep, __ = self.current_position.match_to_catalog_sky(source_pos)
        if cm_sep <= max_sep:
            return source_list[cm_idx]
        

    
class obs_frame:
    def __init__(self, file_name, working_directory=None):
        self.sources = []
        self.source_positions = None
        self.file_name = file_name
        self.time_stamp = None
        self.__update_frame()
        self.__get_working_directory()
        self.__get_beam()
        self.cropped_filepath= None
        self.cropped_wcs = None
        return
    
    def __update_frame(self):
        f = fits.open(self.file_name)[0]
        self.header = f.header
        self.data = f.data
        self.wcs = WCS(self.header, naxis=2)
        self.wcs._naxis = [self.wcs._naxis[0],self. wcs._naxis[1]]
        self.timestamp = self.header['DATE-OBS']
        f.close()
        return 
    
    def __get_working_directory(self, working_directory):
        if working_directory is not None:
            self.working_directory = working_directory
        elif working_directory is None:
            self.working_directory = os.getcwd()
            raise Warning(f"Working directory not specified, taking it to be {self.working_directory}")
        return
    
    def __get_beam(self):
        if "BMAJ" in list(self.header.keys()):
            pix_size = self.header["CDELT2"]
            self.beam = beam(self.header["BMAJ"]/pix_size, self.header["BMIN"]/pix_size, self.header["BPA"])
        else:
            try:
                popt = get_beam_shape_params_iter(self.file_name)
                gauss_axs = popt[3:5]
                maj_ax = gauss_axs.max()
                min_ax = gauss_axs.min()
                theta = theta_to_bpa(popt[5])
                self.beam = beam(maj_ax, min_ax, theta)
            except:
                Warning("Could not get beam shape parameters")
                self.beam = None
        return
    
    def crop_frame(self, center:SkyCoord, dimension: "pix" = 100, out_subdir=None, overwrite=True):
        if out_subdir is None:
            out_subdir = 'cropped_frames/'
        out = os.path.join(self.working_dir, out_subdir, '')
        if not os.path.isdir(out):
            try:
                os.mkdir(out)
            except:
                print(f"Could not make directory {out}")
        sub_frame = Cutout2D(self.data[0,0,:,:], center, dimension, self.wcs)
        out_name = os.path.join(out, self.file_name.split('/')[-1].replace('.fits', '_cropped.fits'))
        fits.writeto(out_name, sub_frame.data, sub_frame.wcs.to_header(), overwrite=overwrite)
        self.cropped_filepath = out_name
        self.cropped_wcs = sub_frame.wcs
        return

    def get_source_positions(self, sigma_threshold=4, cropped=True, verbose=False):
        if cropped:
            fn = self.cropped_filepath
            f = fits.open(fn)[0]
            dat = f.data
            w = self.cropped_wcs
        elif not cropped:
            fn = self.file_name
            f = fits.open(fn)[0]
            dat = f.data[0,0,:,:]
            w = self.wcs
            
        dao = DAOStarFinder(sigma_threshold*rms(dat) + np.nanmedian(dat), fwhm=self.beam.bmaj*2, ratio=self.beam.bmin/self.beam.bmaj, theta=self.beam.bpa)
        sources = dao(dat)
        if len(sources) != 0:
            sources_clip = sources[sources['peak'] > sigma_threshold*rms(dat) + np.nanmedian(dat) ]
            source_clip_sc = w.pixel_to_world(sources_clip['xcentroid'], sources_clip['ycentroid'])
            self.source_positions = source_clip_sc
        elif len(sources) == 0:
            if verbose:
                print(f"No sources found in {fn}")
        return
    

    

class observation:
    def __init__(self, source, full_frame_fns, out_dir, max_sep=0.2*un.deg):
        assert(type(source) == str or type(source) == SkyCoord)
        if type(source) == str:
            # if the source was given as a name, try to resolve in Simbad and apply proper-motion correction
            try:
                source = AccessSIMBAD(source)
                self.source = source
            except Exception as e:
                raise Exception(f"Could not access {source} coordinates through SIMBAD: {e}")
        
        if os.path.isdir(out_dir):
            self.out_dir = out_dir
        elif not os.path.isdir(out_dir):
            raise Exception(f"{out_dir} is not an existing directory")
            
        self.full_frame_fns = full_frame_fns
        self.__init_frames(full_frame_fns, out_dir)
        
        self.max_sep = max_sep
        self.sources = []
        self.latest_idx_w_source = None
        self.equinox = 'fk5'
        return
    
    def __init_frames(self, fns, out_dir):
        frames = []
        for fn in fns:
            frames.append(obs_frame(fn, out_dir))
        self.frames = frames
        return
    
    def crop_frames(self, idx:int = None, dimension=100, out_subdir=None, overwrite=True):
        if idx is None:
            for frame in self.frames:
                frame.crop_frame(self.source, dimension, out_subdir, overwrite)
        elif idx is not None:
            self.frames[idx].crop_frame(self.source, dimension, out_subdir, overwrite)
        return
    
    def find_sources(self, idx:int = None, sigma_threshold=4, cropped=True, verbose=False):
        if idx is None:
            for frame in self.frames:
                frame.get_source_positions(sigma_threshold=sigma_threshold, cropped=cropped, verbose=verbose)
        elif idx is not None:
            self.frames[idx].get_source_positions(sigma_threshold=sigma_threshold, cropped=cropped, verbose=verbose)
        return
    
    def make_reference_frame(self):
        obs_sources = []
        latest_idx_w_source = 0
        while len(obs_sources) == 0:
            if self.frames[latest_idx_w_source].source_positions is not None:
                for j,s in enumerate(self.frames[0].source_positions):
                    source_name = f'source_{str(j+1).zfill(3)}'
                    source = radio_source(label=source_name, frames=np.array([latest_idx_w_source]), avg_position=s, current_position=s)
                    obs_sources.append(source)
                    self.frames[latest_idx_w_source].sources.append(source)
            else:
                latest_idx_w_source += 1
        self.latest_idx_w_source = latest_idx_w_source
        self.sources = obs_sources
        return
    
    def crossmatch_frames(self, ref_frame:obs_frame, new_frame:obs_frame, max_sep=None):
        if max_sep is None:
            max_sep = self.max_sep
            
        if self.sources is None:
            self.make_reference_frame()
        
        source_dict = {}
        ref_sources = ref_frame.sources
        ref_positions = SkyCoord([s.position for s in ref_sources])
        new_positions = SkyCoord(new_frame.positions)
        
        ref_idxs, cm_seps, __ = new_positions.cross_match(ref_positions)
        nearby_sources = ref_sources[ref_idxs[cm_seps<=max_sep]]
        far_sources = ref_sources[ref_idxs[cm_seps>max_sep]]
        
        # need to identify sources that are cross matched to the same reference position
        
        
        
        
        
        return
    
    
  
                
        
#%%
class lightcurve_extraction:
    def __init__(self, source, full_frame_fns: list, out_dir: str, obs_time=None):

        # come from find_associated_sources():
        self.beam_shape_flux_threshold = 8
        self.fwhm = 8
        self.frame_source_dict = {}
        self.frame_source_dict.update({f"frame_{str(i).zfill(3)}":{} for i in range(len(self.full_frame_fns))}) # dictionary of labelled sources per frame
        self.max_sep = 0.2
        self.source_dict = None
        

        
        # comes from find_persistent_sources():
        self.n_source = None
        self.persistent_sources = None # dictionary of sources that showed up persistently in the frames, along with the frame and position
        
        # comes from get_star_positions
        self.star_positions = None
        
        # comes from get_star_fluxes
        self.star_fluxes = None
        self.times = None
        return

    
    def find_associated_sources(self, source_positions=None, max_sep=None):
        """
        Figures out which sources in the self.source_positions property are associated with sources in other frames. Sources are assigned names based on their associations and are specified in the frame_source_dict property.
        :param source_positions: A list of source positions for each of the frames. If None, then it uses the self.source_positions properties. If self.get_source_positions hasn't been run yet, then it runs it with default parameters.
        :type source_positions: np.ndarray of SkyCoord
        :param max_sep: The maximum allowable separations for a source image in two different frames to be considered associated with the same source. If None, it uses the value assigned as self.max_sep
        :type max_sep: float; units degrees
        """
      
        for i in range(len(source_positions)-1):
            # update the latest list of source positions
            frame_key = f'frame_{str(i).zfill(3)}'
            sources = []
            for k in source_list.keys():
                if 'mean_pos' in k:
                    sources.append(source_list[k])
            sources = SkyCoord(sources)

            # before finding sources in the new frame, make sure to only use "actual" sources from the previous frame for reference
            if type(self.source_positions[i]) == SkyCoord:
                latest_idx_w_source = i
                
            ref_key = f'frame_{str(latest_idx_w_source).zfill(3)}'
            ref = self.frame_source_dict[ref_key]['positions']
            ref = SkyCoord(ref)
            
            chk = source_positions[i+1]
            
            # if there were NO sources found in the i+1 frame
            if type(chk) != SkyCoord:
                new_frame_key = f'frame_{str(i+1).zfill(3)}'
                self.frame_source_dict[new_frame_key].update({'sources':[], 'positions':[], 'separations':[],'pos_angles':[]})
                
            # if there were sources found in the i+1 frame to be cross-matched with the reference frame:
            elif type(chk) == SkyCoord:
                cm_idxs, seps, __ = chk.match_to_catalog_sky(ref)
                bad_idxs = np.where(seps > max_sep*un.deg)[0]

                # check if the far away sources are associated with sources not seen in the previous frame. if not, make a new source item
                s = {} # source dictionary for the i+1 frame
                for idx in bad_idxs:
                    s_far = chk[idx]
                    new_idx, new_sep, __ = s_far.match_to_catalog_sky(sources) # cross-matching source with mean position of all sources found so far
                    count = 0
                    
                    # if the source is near the mean position of a previously-found source, then it qualifies as that source
                    if new_sep <= max_sep*un.deg:
                        key_name = f'source_{str(new_idx).zfill(2)}'
                        s.update({key_name:[s_far]})
                    
                    # if the source is not near the mean position of any previously-found sources, then it is considered a new source
                    elif new_sep > max_sep*un.deg:
                        key_name = f'source_{str(int(len(sources)+count)).zfill(2)}' #new source name
                        s.update({key_name:[s_far]}) # assign source position to new dictonary item
                        count += 1

                # now check for the sources that got cross matched in the previous image
                cm_idxs = np.delete(cm_idxs, bad_idxs)
                seps = np.delete(seps, bad_idxs)
                chk = np.delete(chk, bad_idxs)
                for idxi, idx in enumerate(cm_idxs):
                    # for sources that were cross-matched with a source from the reference frame, make a new dictionary item for it
                    if self.frame_source_dict[ref_key]['sources'][idx] not in list(s.keys()):
                        s.update({self.frame_source_dict[ref_key]['sources'][idx]:[]})
#                     s[self.frame_source_dict[ref_key]['sources'][idx]] += {chk[idxi]}
                    s[self.frame_source_dict[ref_key]['sources'][idx]].append(chk[idxi])

                # get rid of duplicate cross matches
                for k in list(s.keys()):
                    if len(s[k])>1:
                        seps = source_list[f'{k}_mean_pos'].separation(SkyCoord(s[k]))
                        min_idx = np.where(seps == np.min(seps))[0][0]
                        s[k] = [s[k][min_idx]]

                # update source dictionary and assignment list
                sa = []
                new_frame_key = f'frame_{str(i+1).zfill(3)}'
                self.frame_source_dict[new_frame_key].update({'sources':[], 'positions':[], 'separations':[],'pos_angles':[]})
                for key in list(s.keys()):
                    sa.append(key)
                    if key not in source_list:
                        source_list.update({key:[]})
                    source_list[key].append(s[key][0])
                    poss = SkyCoord(source_list[key])
                    source_list[f'{key}_mean_pos'] = SkyCoord(poss.ra.mean(), poss.dec.mean(), frame = self.equinox) 
                    self.frame_source_dict[new_frame_key]['sources'].append(key)
                    self.frame_source_dict[new_frame_key]['positions'].append(s[key][0])
                    self.frame_source_dict[new_frame_key]['separations'].append(None)
                    self.frame_source_dict[new_frame_key]['pos_angles'].append(None)
            
        self.max_sep = max_sep
        self.source_dict = source_list
        return
    
    def find_persistent_sources(self, n_source:int = None):
        """
        Identifies which sources show up most persistently. Updates the self.persistent_sources property
        :param n_source: The number of persistent sources in the FOV. If None, then n_source is the median number of sources identified. Default is None
        :type n_source: int
        """
        
        if self.source_dict is None:
            raise warnings.warn("Source dictionary hasn't been produced yet. Running find_associated_sources first.")
        
        if n_source is None:
            n_sources = [len(self.source_positions[i]) for i in range(len(self.crop_fns)) if type(self.source_positions[i]) == SkyCoord]
            n_source = int(np.median(n_sources))
        self.n_source = n_source
        
        source_list_lens = []
        keys = []
        for key in list(self.source_dict.keys()):
            if 'mean_pos' not in key:
                source_list_lens.append(len(self.source_dict[key]))
                keys.append(key)

        source_list_lens, keys = zip(*sorted(zip(source_list_lens, keys), reverse=True)) # sort sources by the number of times they were identified
        keys_final = keys[:n_source]
        
        persistent_sources = {}
        for key in keys_final:
            persistent_sources.update({key:self.source_dict[f'{key}_mean_pos']})
        
        self.persistent_sources = persistent_sources
        return
    
    def find_separation_and_pos_angs(self):
        """
        Finds the separation and position angle between persistently detected sources and their mean position. Informative for understanding shifts due to the ionosphere. Updates the 'separations' and 'pos_angles' values in the frames of the self.frame_source_dict property
        """
        if self.persistent_sources is None:
            self.find_persistent_sources()
            
        for i,sources in enumerate(self.source_positions):
            frame_key = f"frame_{str(i).zfill(3)}"
            sa = self.frame_source_dict[frame_key]["sources"]
            source_keys = [k for k in sa if k in self.persistent_sources.keys()]
            for k in source_keys:
                idx = self.frame_source_dict[frame_key]['sources'].index(k)
                mean_pos = self.persistent_sources[k]
                sep = mean_pos.separation(self.frame_source_dict[frame_key]['positions'][idx])
                pos_ang = mean_pos.position_angle(self.frame_source_dict[frame_key]['positions'][idx])
                self.frame_source_dict[frame_key]['separations'][idx] = sep.value
                self.frame_source_dict[frame_key]['pos_angles'][idx] = pos_ang.value
        return
    
    def get_source_flux(self, source_keys, frame_key:str = None, frame_idx:int = None, get_bkg_rms=False):
        """
        gets the flux for all sources indicated by source_keys for a given frame
        :param source_keys: the name of the source to extract flux from
        :type source_keys: str
        :param frame_key: the the name of the frame in the frame_source_dict property to reference for extracting source flux
        :type frame_key: str
        :param frame_idx: alternative to frame_key parameter-- the index of the frame to reference for extracting source flux
        :type frame_idx: int
        :param get_bkg_rms: if True, returns the source-subtracted noise of the frame that the flux was extracted from
        :type get_bkg_rms: bool
        :return flux, [bkg_rms]: the flux of the source and, if get_bkg_rms is true, the noise of the source-subtracted frame
        :rtype: float, [float]
        """
        assert((frame_key != None) or (frame_idx != None)), "Either frame_key or frame_idx needs to be defined"
        if frame_key is not None:
            assert(frame_key in self.frame_source_dict.keys()), f"Frame key {frame_key} not recognized"
            frame_idx = int(frame_key.split("_")[-1])
               
        elif frame_idx is not None:
            frame_key = f"frame_{str(frame_idx).zfill(3)}"
            
        if type(source_keys) is str:
            source_keys = [source_keys]
        
        flux = np.zeros(len(source_keys))
        
        f = fits.open(self.crop_fns[frame_idx])[0]
        hdr = f.header
        dat = f.data[0,0,:,:]
        w = WCS(hdr, naxis=2)
        for i, k in enumerate(source_keys):
            if k in self.frame_source_dict[frame_key]['sources']:
                idx = self.frame_source_dict[frame_key]['sources'].index(k)
                position = self.frame_source_dict[frame_key]['positions'][idx]

                x,y = w.world_to_pixel(position)
                x = int(np.round(x))
                y = int(np.round(y))
                flux[i] = dat[y,x]

            elif k not in self.frame_source_dict[frame_key]['sources']:
                flux[i] = np.nan
                
        if get_bkg_rms:
            dat_mask = np.zeros(dat.shape)
            a = hdr['BMAJ']/hdr['CDELT2']
            b = hdr['BMIN']/hdr['CDELT2']
            theta = np.pi/2 + hdr['BPA']*un.deg.to('rad')
            for idx, k in enumerate(self.frame_source_dict[frame_key]['sources']):
                ap = EllipticalAperture(w.world_to_pixel(frame['positions'][idx]),a,b,theta=theta)
                mask = ap.to_mask()
                try:
                    dat_mask[mask.bbox.iymin:mask.bbox.iymax,mask.bbox.ixmin:mask.bbox.ixmax] = mask.data
                except:
                    pass
            dat_masked = np.ma.masked_array(dat, dat_mask)
            bkg_rms[i] = np.sqrt(np.mean(dat_masked**2))      
            return flux, bkg_rms
        
        return flux
    
    def get_single_source_fluxes(self, source_key:str):
        """
        gets the fluxes for a single source across all frames
        :param source_key: the name of the source to extract flux for
        :type source_key: str
        :return: fluxes for the source
        :rtype: np.ndarray of floats
        """
        frame_keys = self.frame_source_dict.keys()
        fluxes = np.zeros(len(frame_keys))
        for i, k in enumerate(frame_keys):
            fluxes[i] = self.get_source_flux(source_key, frame_key = k)[0]
        return fluxes
    
    def get_all_source_fluxes(self):
        """
        Gets fluxes for all persistent_sources across all frames
        """
        frame_keys = self.frame_source_dict.keys()
        source_keys = self.persistent_sources.keys()
        fluxes = np.zeros((len(frame_keys), len(source_keys)))
        for i, fk in enumerate(frame_keys):
            fluxes[i] = self.get_source_flux(source_keys, frame_key = fk)
        return fluxes
    
    def make_position_change_map(self, frame_idx, delta_ra=0.25*un.deg, delta_dec=0.25*un.deg, n_steps=1, n_iter=3, n_avg=3, plot=True, cmap='hsv'):
        source_keys = self.persistent_sources.keys()
        frame_key = f"frame_{str(frame_idx).zfill(3)}"
        
        source_positions = []
        source_separations = []
        source_pos_angles = []
        frame = self.frame_source_dict[frame_key]
        for j,sk in enumerate(source_keys):
            if sk in frame['sources']:
                idx = frame['sources'].index(sk)
                source_positions.append(frame['positions'][idx])
                source_separations.append(frame['separations'][idx])
                source_pos_angles.append(frame['pos_angles'][idx])
            elif sk not in frame['sources']:
                source_positions.append(SkyCoord(ra=0*un.deg, dec=0*un.deg, frame='fk5'))
                source_separations.append(np.nan)
                source_pos_angles.append(np.nan)
                
        seps = np.array(source_separations)
        pos_angles = np.array(source_pos_angles)
        coords = SkyCoord(source_positions)
        ras = coords.ra
        decs = coords.dec
        
        hdr = fits.open(self.crop_fns[frame_idx])[0].header
        w = WCS(hdr, naxis=2)
        min_pos = w.pixel_to_world(w.pixel_shape[0],0)
        max_pos = w.pixel_to_world(0,w.pixel_shape[1])
        ra_min = min_pos.ra
        ra_max = max_pos.ra
        dec_min = min_pos.dec
        dec_max = max_pos.dec
        
        n_elements_dec = int((dec_max - dec_min)/delta_dec)
        n_elements_ra = int((ra_max - ra_min)*np.cos((dec_max + dec_min)/2)/delta_ra)
        #delta_ra = delta_ra * np.cos((dec_max + dec_min)/2)
        delta_ra = delta_ra/np.cos((dec_max + dec_min)/2)
        #print(delta_ra)
        grid_pos_angles = np.zeros((n_elements_ra, n_elements_dec))
        grid_seps = np.zeros((n_elements_ra, n_elements_dec))
        for r in range(n_elements_ra):
            if r == 0:
                ra0 = ra_min
            raf = ra0 + delta_ra
            dec0 = dec_min
            idxs_ra = np.where((ras >= ra0 - n_steps*delta_ra) & (ras <= raf+ n_steps*delta_ra))[0]
            
            for d in range(n_elements_dec):
                decf = dec0 + delta_dec
                if len(idxs_ra) != 0:
                    ra_select_ra = ras[idxs_ra]
                    ra_select_dec = decs[idxs_ra]
                    idxs = np.where((ra_select_dec >=dec0 - n_steps*delta_dec) &(ra_select_dec<=decf+n_steps*delta_dec))[0]
                    if len(idxs) == 0:
                        sep_avg = 0
                        pos_ang_avg = 0
                    elif len(idxs) != 0:
                        true_idxs = []
                        for i in idxs:
                            true_idxs.append(np.where(decs == ra_select_dec[i])[0][0])
                        sep_avg = np.nanmean(seps[true_idxs])
                        pos_ang_avg = np.nanmean(pos_angles[true_idxs])
                    grid_seps[r,d] = sep_avg
                    grid_pos_angles[r,d] = pos_ang_avg
                elif len(idxs_ra) == 0:
                    grid_seps[r,d] = 0
                    grid_pos_angles[r,d] = 0
                dec0 = decf
            ra0 = raf
        
        count = 0
        while count < n_iter:
            grid_seps = uniform_filter(grid_seps, n_avg)
            grid_pos_angles = uniform_filter(grid_pos_angles, n_avg)
            count += 1
        if plot:
            hsv = mp.cm.get_cmap(cmap)
            fig, ax = plt.subplots()
            fig.set_figwidth(5)
            fig.set_figheight(5)
            for r in range(n_elements_ra):
                for d in range(n_elements_dec):
                    color = hsv(grid_pos_angles[r,d]/(2*np.pi))
                    color = adjust_lightness(color, grid_seps[r,d]/np.max(grid_seps))
                    patch = patches.Rectangle((r,d),1,1, color = color)
                    ax.add_patch(patch)
            ax.set_xlim([0,n_elements_ra])
            ax.set_ylim([0,n_elements_dec])
            plt.show()
            return grid_seps, grid_pos_angles, fig, ax 
        return grid_seps, grid_pos_angles
    
    def find_weighted_position_change(self, frame_key:str = None, frame_idx:int = None):
        """
        Gets the weighted position *change* of the star of interest given its astronomical position separation from the persistently-detected sources.
        :param frame_key: the key for the frame in frame_source_dict to get the weighted position change
        :type frame_key: str
        :param frame_idx: alternative to frame_key-- the index for the frame in frame_source_dict to get the weighted position change
        :type frame_idx: int
        :return: sep, theta-- the separation and position angle between assigned coordinate and where its changed to
        :rtype: float*astropy.Unit.deg, float*astropy.Unit.rad
        """
        assert((frame_key != None) or (frame_idx != None)), "Either frame_key or frame_idx needs to be defined"
        if frame_key is not None:
            assert(frame_key in self.frame_source_dict.keys()), f"Frame key {frame_key} not recognized"
            frame_idx = int(frame_key.split("_")[-1]) 
        elif frame_idx is not None:
            frame_key = f"frame_{str(frame_idx).zfill(3)}"
        
        pi2 = np.pi/2 * un.rad
        source_star_separations = []
        position_vectors = []
        frame = self.frame_source_dict[frame_key]
        
        for k in self.persistent_sources.keys():
            if k in frame['sources']:
                idx = frame['sources'].index(k)
                pos = frame['positions'][idx]
                ppos = self.persistent_sources[k]
                
                mag = ppos.separation(pos).value
                ang = -ppos.position_angle(pos) + 5*pi2
                
                position_vectors.append([mag*np.cos(ang), mag*np.sin(ang)])
                source_star_separations.append(ppos.separation(self.source).value)
                
        if len(source_star_separations) != 0:
            source_star_separations = np.array(source_star_separations)
            weights = source_star_separations.sum()/source_star_separations
            weights_norm = weights/np.sum(weights**2)**0.5

            vecs_weight = (np.array(position_vectors).transpose()*weights_norm).transpose()
            vecs_sum = vecs_weight.mean(axis=0)
            sep = (vecs_sum**2).sum()**0.5
            theta = np.arctan(vecs_sum[1]/vecs_sum[0])*un.rad
            if theta < 0:
                theta += 2*np.pi*un.rad
            theta = -theta + 5*pi2
        elif len(source_star_separations) == 0:
            warnings.warn(f"No sources found in the field to estimate position offset for frame {frame_key}")
            sep = 0
            theta = 0*un.rad
        return sep*un.deg, theta
    
    def get_star_position(self, frame_key):
        """
        Gets the new position of the star/coordinate based on weighted position change found for a given frame
        :param frame_key: the key for the frame in frame_source_dict to get the new star/coordinate position
        :type frame_key: str
        :return fixed_coord: new coordinate of the star/coordinate
        :rtype fixed_coord: astropy.coordinates.SkyCoord
        """
        sep, theta = self.find_weighted_position_change(frame_key=frame_key)
        fixed_coord = self.source.directional_offset_by(theta, sep)
        return fixed_coord
    
    def get_star_positions(self):
        """
        Gets the position of the star for all frames and update self.star_positions.
        """
        fks = self.frame_source_dict.keys()
        positions = []
        for k in fks:
            fixed_coord = self.get_star_position(frame_key=k)
            positions.append(fixed_coord)
        self.star_positions = SkyCoord(positions)
        return
    
    def get_star_fluxes(self):
        if self.star_positions is None:
            print("Finding star positions first")
            self.get_star_positions()
            
        pix_flux = np.zeros(len(self.crop_fns))
        for i, fn in enumerate(self.crop_fns):
            f = fits.open(fn)[0]
            hdr = f.header
            dat = f.data[0,0,:,:]
            w = WCS(hdr, naxis = 2)
            
            x,y = w.world_to_pixel(self.star_positions[i])
            x = int(np.round(x))
            y = int(np.round(y))
            pix_flux[i] = dat[y,x]
        self.star_fluxes = pix_flux
        return
    
    def save_data(self, outn=None):
        if outn is None:
            outn = os.path.join(self.out_dir,'')
            outn = f"{outn}lightcurve_extraction_class_data.npz"
        np.savez(outn, self.__dict__)
        return
    
def load_data(fp):
    """
    loads in a lightcurve_extraction
    :param fp: the filepath and name of the file resulting from lightcurve_extraction.save_data
    :type fp: str
    """
    d = np.load(fp, allow_pickle=True)['arr_0'].flatten()[0]
    lc_ext = lightcurve_extraction(source=d['source'], full_frame_fns=d['full_frame_fns'], out_dir=d['out_dir'], obs_time=d['obs_time'])
    for k in d.keys():
        lc_ext.__dict__[k] = d[k]
    return lc_ext


#     def return_source_fluxes(self, source_key):
#         maxvs = np.zeros(len(self.frame_source_dict.keys()))
#         minvs = np.zeros(len(self.frame_source_dict.keys()))
#         means = np.zeros(len(self.frame_source_dict.keys()))
#         medians = np.zeros(len(self.frame_source_dict.keys()))
#         stds = np.zeros(len(self.frame_source_dict.keys()))
#         for i,k in enumerate(list(self.frame_source_dict.keys())):
#             f = fits.open(self.crop_fns[i])[0]
#             data = f.data[0,0,:,:]
#             hdr = f.header
#             w = WCS(hdr, naxis=2)
#             try:
#                 a = hdr['BMAJ']/hdr['CDELT2']
#                 b = hdr['BMIN']/hdr['CDELT2']
#                 theta = np.pi/2 + hdr['BPA']*un.deg.to('rad')
#             except:
#                 print("Could not get beam parameters. Using circular aperture instead")
#                 a = self.fwhm
#                 b = self.fwhm
#                 theta = 0
#             try:
#                 idx = self.frame_source_dict[f'frame_{str(i).zfill(3)}']['sources'].index(source_key)
#                 ap = EllipticalAperture(w.world_to_pixel(self.frame_source_dict[f'frame_{str(i).zfill(3)}']['positions'][idx]),a,b,theta=theta)
#                 m = ap.to_mask()
#                 dm = m.multiply(data)
#                 maxvs[i] = np.nanmax(dm)
#                 minvs[i] = np.nanmin(dm)
#                 medians[i] = np.nanmedian(dm)
#                 means[i] = np.nanmean(dm)
#                 stds[i] = np.nanstd(dm)
#             except:
#                 print(f"Could not get flux for source {source_key} on frame {k}")
            
#         return maxvs, minvs, medians, means, stds
        
            
            