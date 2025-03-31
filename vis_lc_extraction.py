import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.ndimage.filters import uniform_filter

from astropy.io import fits
from astropy import constants as const, units as un
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord,EarthLocation, get_sun, get_moon, AltAz, SkyCoord
from astroquery.simbad import Simbad

from photutils.aperture import CircularAperture, aperture_photometry
from photutils import DAOStarFinder, EllipticalAperture

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

def get_timestamp(fn, return_type=str):
    """
    Get timestamp of integration based on filename.
    :param fn: filename to get timestamp from
    :type fn: str
    :param return_type: The return type of the timestamp. Either str or Time; default is str
    :type return_type: str or Time
    :return: timestamp of integration
    :rtype: str or Time
    """
    base = fn.split('/')[-1] # remove the subdirectories and only look at the actual file's name
    ts = base.split('.')[0] # the timestamp is the first element in the standard ms naming scheme
    if return_type is str:
        return ts
    elif return_type is Time:
        return Time(ts, format = 'isot')
    return

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


def get_beam_shape_params(dat, dpix=12, update_file_header=True, threshold=8, fwhm=8):
    """
    Gets the beam shape parameters that can be used in twoD_Gaussian to construct a psf. Necessary if the data were not deconvolved. Assumes that the sources are sufficiently circular that DAOStarFinder can identify to extract-- if this doesn't work, consider using get_beam_shape_params_v2
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


def get_beam_shape_params_v2(fn, update_file_header=True, subframe_radius = (50,50), std_threshold = 6, niter = 21, sma=15, eps=0.7, pa=2):
    """
    Gets the beam shape parameters that can be used in twoD_Gaussian to construct a psf. Necessary if the data were not deconvolved. This is slower than get_beam_shape_params, but can identify even extremely elliptical sources.
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


def get_source_positions(fn, thresh_pix=3.5, thresh_isl=2, rms_box=(50,15), min_source_flux=3, fwhm=8, threshold=8, dynamic=True, method='dao'):
    """
    Gets the positions of sources in a FITS file either through DAOStarFinder or pybdsf. I recommend DAOStarFinder
    :param fn: the file name of the FITS file
    :type fn: str
    :param thresh_pix: Only used if method is 'bdsf', in which case it sets the minimum SNR of the brightest pixel of a source island
    :type thresh_pix: float
    :param thresh_isl: Only used if the method is 'bdsf'
    :type thresh_isl: float
    :param rms_box: Only used if the method is 'bdsf'. Defines the box size that is used to estiamte the rms
    :type rms_box: tuple length 2
    :param min_source_flux: Only used if the method is 'bdsf'. Defines the minimum flux of a source for it to be included as a source
    :param fwhm: Only used if the method='dao' and dynamic=False. Sets the full-width half-maximum that is used in DAOStarFinder
    :param threshold: Only used if the method='dao'. Describes the SNR required for detecting a source. Default is 3 (3*std above the median)
    :param dynamic: Defines whether to use a hardcoded value of fwhm (the fwhm param) or to use the FWHM as defined by BMAJ and CDELT2 in the FITS file header. If True, use the FITS header info. If the image is not deconvolved, you need to run get_beam_shape_params with update_file_header=True first. Default is True
    :param method: defines the source-identification method; acceptable inputs are 'bdsf' and 'dao'. If bdsf is used and the file is not deconvolved, you need to run get_beam_shape_params with update_file_header=True first. Default is 'dao'
    :return: The source positions
    :rtype: SkyCoord or None
    """
    if not dynamic:
        fwhm = fwhm
        
    elif dynamic:
        try:
            hdr = fits.open(fn)[0].header
            if hdr['BMAJ'] !=0:
                maj_ax = hdr['BMAJ']/hdr['CDELT2']
                min_ax = hdr['BMIN']/hdr['CDELT2']
                theta = hdr['BPA']
            elif hdr['BMAJ'] == 0:
                fwhm = fwhm
        except:
            fwhm = fwhm
            
    if method.lower() == 'bdsf':
        result = bdsf.process_image(fn, quiet=True, thresh_pix=thresh_pix, thresh_isl=thresh_isl, rms_box=rms_box)
        positions = np.array([src.posn_sky_centroid for src in result.sources if src.peak_flux_max > min_source_flux ])
        source_positions = SkyCoord(positions, unit=(un.deg, un.deg))
        
    elif method.lower() == 'dao':
        f = fits.open(fn)[0]
        hdr = f.header
        dat = f.data[0,0,:,:]
        w = WCS(hdr, naxis=2)
        thresh = np.median(dat) + threshold*np.std(dat)
        
        try:
            # asserting ellipticity is the best, but some versions of DAOStarFinder may not support these parameters
            daofind = DAOStarFinder(threshold=thresh, fwhm=maj_ax*2, ratio=min_ax/maj_ax, theta=theta)
        except:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh)
            
        sources = daofind(dat)
        if sources is not None:
            x = sources['xcentroid']
            y = sources['ycentroid']
            source_positions = w.pixel_to_world(x,y)
            
        elif sources is None:
            source_positions = None

    return source_positions


def AccessSIMBAD(objID:str):
    """
    Returns SkyCoords object of the target including RA, Dec, and proper motions
    :param objID: The name of the target object
    :type objID: str
    :return: The SkyCoord object with RA, Dec, and proper motions
    :rtype: SkyCoord
    """
    try:
        Simbad.add_votable_fields('main_id','propermotions')
        simbadResult = Simbad.query_object(objID)[0]
    except:
        print('Unable to access the object from SIMBAD.')
        return
    
    try:
        dist_val = simbadResult['Distance_distance']
        dist_unit = un.Unit(str(simbadResult['Distance_unit']))
        dist = dist_val*dist_unit
        
    except:
        warnings.warn('Unable to determine distance of object '+objID)
        dist = 100*un.pc
        
    obj_ra_str = simbadResult['RA'] 
    obj_dec_str = simbadResult['DEC']
    obj_ra_mu = simbadResult['PMRA'] * un.mas/un.yr
    obj_dec_mu = simbadResult['PMDEC'] * un.mas/un.yr
    obs_time = "J2000"
    coord_str = obj_ra_str + ' ' + obj_dec_str
    coords = SkyCoord(coord_str,distance = dist,frame = 'icrs', unit = (un.hourangle, un.deg), obstime = obs_time, pm_ra_cosdec = obj_ra_mu, pm_dec = obj_dec_mu)
    
    return coords


def ApplyProperMotion(obj:SkyCoord,obstime: Time):
    """
    Applies proper motion to get current position of target. SkyCoords object must include proper motion information
    :param obj: SkyCoord object of the target to apply proper motion to. Must include proper motion information
    :type obj: SkyCoord
    :param obstime: Time object of the observing date for which you want to know 
    :type obstime: Time
    :raises Exception: Raised when it cannot apply the proper motion calculation, likely due to error in observation time or lack of proper motion information
    :return: New coordinates of the object accounting for proper motion since J2000
    :rtype: SkyCoord
    """
    try:
        new_coordinates = obj.apply_space_motion(new_obstime = obstime)
        return new_coordinates
    except:
        raise Exception('Unable to apply proper motion')
        

class lightcurve_extraction:
    def __init__(self, source, full_frame_fns: list, out_dir: str, obs_time=None):
        """
        Class for extracting light curves for sources 
        :param source: either the name or the SkyCoord coordinates of a source. If a name, it will try to resolve the source using AccessSimbad and the obs_time parameter. If a SkyCoord coordinate, assumes that the position is already proper motion corrected.
        :type source: str or astropy.coordinates.SkyCoord
        :param full_frame_fns: A list of the file names of the full-frame images
        :type full_frame_fns: list of strings
        :param out_dir: The directory that cropped files and solutions should be saved to
        :type out_dir: str
        :param obs_time: the general time of the observation. If the source given is a name, it will use this to correct the source position due to proper motion
        :type obs_time: astropy.time.Time or str
        """
        
        assert(type(source) == str or type(source) == SkyCoord)
        self.obs_time = obs_time
        
        if type(obs_time) == str:
            self.obs_time = Time(obs_time,format='isot')
            
        if type(source) == str:
            # if the source was given as a name, try to resolve in Simbad and apply proper-motion correction
            try:
                source = AccessSIMBAD(source)
                source = ApplyProperMotion(source, self.obs_time)
                self.source = source
            except Exception as e:
                raise Exception(f"Could not access {source} coordinates through SIMBAD: {e}")
                
        elif type(source) == SkyCoord:
            self.source = source
        
        self.full_frame_fns = full_frame_fns
        if os.path.isdir(out_dir):
            self.out_dir = out_dir
        elif not os.path.isdir(out_dir):
            raise Exception(f"{out_dir} is not an existing directory")
        
        # come from find_associated_sources():
        self.beam_shape_flux_threshold = 8
        self.fwhm = 8
        self.frame_source_dict = {}
        self.frame_source_dict.update({f"frame_{str(i).zfill(3)}":{} for i in range(len(self.full_frame_fns))}) # dictionary of labelled sources per frame
        self.max_sep = 0.2
        self.source_dict = None
        
        # come from crop_fits():
        self.equinox = 'fk5'
        self.crop_path = None # the path to the cropped-data directory
        self.crop_fns = None # the names of the cropped fits files
        self.crop_dimensions = None # the dimensions of the cropped images
        
        # comes from get_source_positions():
        self.source_positions = None # array of source positions in each frames
        self.thresh_pix = 3.5
        self.thresh_isl = 2
        self.rms_box = (50,15)
        self.min_source_flux = 3
        
        # comes from find_persistent_sources():
        self.n_source = None
        self.persistent_sources = None # dictionary of sources that showed up persistently in the frames, along with the frame and position
        
        # comes from get_star_positions
        self.star_positions = None
        
        # comes from get_star_fluxes
        self.star_fluxes = None
        self.times = None
        return
    
    def get_timestamps(self, do_return=True):
        """
        gets the integration timestamps for all of the files
        :param do_return: if True, returns the times as a list in addition to updating the self.times property. Default is True
        :param do_return: bool
        :return: list of times
        :rtype: list of astropy.time.Time objects
        """
        times = []
        for fn in self.full_frame_fns:
            times.append(get_timestamp(fn, return_type=Time))
        self.times = times
        if do_return:
            return times
    
    def crop_fits(self, out_subdir:str = None, dim_pix:int = 100, save=True, overwrite=True):
        """
        crops all fits files with the source of interest in the center pixel.
        :param out_subdir: the directory under out_dir that cropped files will be saved to
        :type out_subdir: str
        :param dim_pix: the x, y dimension of the cropped image; makes a cropped image of dimensions 2*dim_pix x 2*dim_pix. Default is 100
        :type dim_pix: int
        :param save: if True, it will save a new file with the cropped data and new header information. Default is True
        :type save: bool
        :param overwrite: For if you want to overwrite the header of cropped files of the same file name. Default is True
        :type overwrite: bool
        :return: None
        """
        # Make the subdirectory for the cropped files:
        if out_subdir is None:
            out_subdir = 'cropped_frames/'
        out = os.path.join(self.out_dir, out_subdir, '')
        if not os.path.isdir(out):
            try:
                os.mkdir(out)
            except:
                print(f"Could not make directory {out}")
                
        subframes = np.zeros([len(self.full_frame_fns), int(2*dim_pix), int(2*dim_pix)])
        timestamps = []
        for i, fn in enumerate(self.full_frame_fns):
            try:
                f = fits.open(fn)[0]
                hdr = f.header
                w = WCS(hdr, naxis = 2)
                y, x = w.world_to_pixel(self.source)
                x = int(np.round(x,0))
                y = int(np.round(y,0))
                subframe = f.data[0, 0, x-dim_pix:x+dim_pix, y-dim_pix:y+dim_pix]
                subframes[i] = subframe
                timestamps.append(Time(f.header['DATE-OBS'], format='isot').mjd)
                if save:
                    hdr['CRVAL1'] = self.source.ra.value
                    hdr['CRVAL2'] = self.source.dec.value
                    hdr['CRPIX1'] = dim_pix + 1
                    hdr['CRPIX2'] = dim_pix + 1

                    outn = f"{out}{fn.split('/')[-1].replace('.fits', '_cropped.fits')}"
                    fits.writeto(outn, f.data[:, :, x-dim_pix:x+dim_pix, y-dim_pix:y+dim_pix], hdr, overwrite=overwrite)
                    
            except Exception as e:
                print(f"Failed on {i}: {e}")
                
        crop_fns = glob.glob(f"{out}*fits")
        crop_fns.sort()
        self.crop_path = out
        self.crop_fns = crop_fns
        self.crop_dimensions = subframe.shape
        self.times = timestamps
        return
    
    def get_beam_shape_params(self, idx: int = None, dpix=12, update_file_header=True, fwhm=None, threshold=None, beam_shape_ver = "1"):
        """
        gets the beam shape parameters for all of the files. Necessary if the data are not deconvolved
        :param idx: index of the cropped file of interest if the beam parameters of only one file is wanted
        :type idx: int
        :param dpix: value for dpix in get_beam_shape_params
        """
        if fwhm is None:
            fwhm = self.fwhm
        if threshold is None:
            threshold = self.beam_shape_flux_threshold
            
        if idx is None:
            popt_vals = np.zeros((len(self.crop_fns),7))
            for i, fn in enumerate(self.crop_fns):
                popt_vals[i] = get_beam_shape_params(fn, dpix, update_file_header, threshold, fwhm)
                
        elif idx is not None:
            popt_vals = get_beam_shape_params(self.crop_fns[idx], dpix, update_file_header, threshold, fwhm)
        return popt_vals
    
    def get_source_positions(self, idx:int = None, thresh_pix=None, thresh_isl=None, rms_box=None, min_source_flux=None, threshold=8, fwhm=8, dynamic=True, method='dao', return_res=False):
        """
        gets the positions of sources detectable in a given frame (if idx is specified) or in all frames if idx is not specified. These get saved to the source_positions property as a list of SkyCoord lists.
        :param idx: indicates the index of the cropped_fn list to get source positions for. If None, it does this for all images in the cropped_fn list. Default is None
        :type index: int
        :param thresh_pix: Only used if the method is 'bdsf', which is not recommended
        :type thresh_pix: int
        :param thresh_isl: Only used if the method is 'bdsf', which is not recommended
        :type thresh_isl: float
        :param rms_box: Only used if the method is 'bdsf', which is not recommended
        :type rms_box: tuple of len 2
        :param min_source_flux: minimum flux an identified source needs to be included in the final list
        :type min_source_flux: float
        :param threshold: Only used if the method is 'dao'
        :type threshold: float
        :param fwhm: Only used if the method is 'dao' and dynamic=False
        :type fwhm: float
        :param dynamic: if True, then the beam shape changes based on beam parameters in the file header, otherwise it uses the fwhm parameter to define the beam shape. Default is True
        :type dynamic: bool
        :param method: the method to use to look for sources, either pybdsf ('bdsf') or DAOStarfinder ('dao'). Default is 'dao'
        :type method: str
        :param return_res: return the results in addition to assigning them to the self.source_positions property. Default is False (only assigns to self.source_positions)
        :type return_res: bool
        :return: If return_res=True, returns a numpy array of SkyCoord lists-- each array index is the list of sources for a frame of that same index
        :rtype: np.ndarray
        """
        if thresh_pix is None:
            thresh_pix = self.thresh_pix
        if thresh_isl is None:
            thresh_isl = self.thresh_isl
        if rms_box is None:
            rms_box = self.rms_box
        if min_source_flux is None:
            min_source_flux  = self.min_source_flux
            
        if idx is None:
            source_positions = np.zeros(len(self.crop_fns), dtype=object)
            for i,fn in enumerate(self.crop_fns):
                sp = get_source_positions(fn, thresh_pix, thresh_isl, rms_box, min_source_flux, fwhm=fwhm, threshold=threshold, dynamic=dynamic, method=method)
                if sp is not None:
                    source_positions[i] = sp
        
        elif idx is not None:
            source_positions = get_source_positions(self.crop_fns[idx], thresh_pix, thresh_isl, rms_box, min_source_flux, fwhm=fwhm, threshold=threshold, dynamic=dynamic, method=method)
            return source_positions
        
        self.thresh_pix = thresh_pix
        self.thresh_isl = thresh_isl
        self.rms_box = rms_box
        self.min_source_flux = min_source_flux
        self.source_positions = source_positions 
        if return_res:
            return source_positions
    
    def find_associated_sources(self, source_positions=None, max_sep=None):
        """
        Figures out which sources in the self.source_positions property are associated with sources in other frames. Sources are assigned names based on their associations and are specified in the frame_source_dict property.
        :param source_positions: A list of source positions for each of the frames. If None, then it uses the self.source_positions properties. If self.get_source_positions hasn't been run yet, then it runs it with default parameters.
        :type source_positions: np.ndarray of SkyCoord
        :param max_sep: The maximum allowable separations for a source image in two different frames to be considered associated with the same source. If None, it uses the value assigned as self.max_sep
        :type max_sep: float; units degrees
        """
        
        if max_sep is None:
            max_sep = self.max_sep
            
        if source_positions is None:
            if self.source_positions is None:
                source_positions = self.get_source_positions()
            elif self.source_positions is not None:
                source_positions = self.source_positions
                
        assert(len(source_positions) == len(self.crop_fns)) # makes sure the number of source list indices matches the number of images
        
        source_list = {} #dictionary of all sources found across the frames
        self.frame_source_dict['frame_000'].update({'sources':[], 'positions':[], 'separations':[],'pos_angles':[]}) # initialize first frame
        
        # update source information for the first frame. Note: assumes that there were sources detected in the first frame
        for j,s in enumerate(source_positions[0]):
                source_list.update({f'source_{str(j).zfill(2)}':[s]})
                source_list.update({f'source_{str(j).zfill(2)}_mean_pos':s}) 
                self.frame_source_dict['frame_000']['sources'].append(f'source_{str(j).zfill(2)}')
                self.frame_source_dict['frame_000']['positions'].append(s)
                self.frame_source_dict['frame_000']['separations'].append(None)
                self.frame_source_dict['frame_000']['pos_angles'].append(None)   
                
        latest_idx_w_source = 0 # the most recent frame index with a source
        
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
        
            
            