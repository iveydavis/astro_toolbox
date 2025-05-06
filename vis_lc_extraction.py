import numpy as np
import scipy.optimize as opt

from astropy.io import fits
from astropy import units as un
from astropy.time import Time
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from photutils import DAOStarFinder, EllipticalAperture

import matplotlib.pyplot as plt

import os
import warnings
import copy


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
        
class background:
    def __init__(self, mean, median, std, rms):
        self.mean = mean
        self.median = median
        self.std = std
        self.rms = rms
        return
    
class obs_frame:
    def __init__(self, file_name, index=None, working_directory=None):
        self.source_positions = None
        self.file_name = file_name
        self.timestamp = None
        self.__update_frame()
        self.__get_working_directory(working_directory)
        self.__get_beam()
        self.cropped_filepath= None
        self.cropped_wcs = None
        self.index = index
        self.background = background(None, None, None, None)
        return
    
    def __update_frame(self):
        f = fits.open(self.file_name)
        self.header = f[0].header
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
            warnings.warn(f"Working directory not specified, taking it to be {self.working_directory}")
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
        out = os.path.join(self.working_directory, out_subdir, '')
        if not os.path.isdir(out):
            try:
                os.mkdir(out)
            except:
                print(f"Could not make directory {out}")
        data = fits.open(self.file_name)[0].data[0,0,:,:]
        sub_frame = Cutout2D(data, center, dimension, self.wcs)
        out_name = os.path.join(out, self.file_name.split('/')[-1].replace('.fits', '_cropped.fits'))
        fits.writeto(out_name, sub_frame.data, sub_frame.wcs.to_header(), overwrite=overwrite)
        self.cropped_filepath = out_name
        self.cropped_wcs = sub_frame.wcs
        return
    
    def get_data_and_wcs(self, cropped):
        if cropped:
            fn = self.cropped_filepath
            f = fits.open(fn)[0]
            if len(f.data.shape) == 4:
                dat = f.data[0,0,:,:]
            else:
                dat = f.data
            w = self.cropped_wcs
            
        elif not cropped:
            fn = self.file_name
            f = fits.open(fn)[0]
            if len(f.data.shape) == 4:
                dat = f.data[0,0,:,:]
            else:
                dat = f.data
            w = self.wcs
        return dat, w, fn
    
    def get_source_positions(self, sigma_threshold=4, cropped=True, verbose=False):
        dat, w, fn = self.get_data_and_wcs(cropped)
            
        dao = DAOStarFinder(threshold=sigma_threshold*rms(dat) + np.nanmedian(dat), fwhm=self.beam.bmaj*2, ratio=self.beam.bmin/self.beam.bmaj, theta=(90 + self.beam.bpa))
        sources = dao(dat)
        if len(sources) != 0:
            # sources_clip = sources[sources['peak'] > sigma_threshold*rms(dat) + np.nanmedian(dat) ]
            # source_clip_sc = w.pixel_to_world(sources_clip['xcentroid'], sources_clip['ycentroid'])
            source_clip_sc = w.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            self.source_positions = source_clip_sc
        elif len(sources) == 0:
            if verbose:
                print(f"No sources found in {fn}")
            self.source_positions = SkyCoord([]*un.deg, []*un.deg)
        return
    
    def get_background_stats(self, cropped=True):
        dat, w, fn = self.get_data_and_wcs(cropped)
        dat_mask = np.zeros(dat.shape)
        
        a = self.beam.bmaj
        b = self.beam.bmin
        theta = self.beam.bpa + 90
        
        sources = self.source_positions
        
        for idx, pos in enumerate(sources):
            ap = EllipticalAperture(w.world_to_pixel(pos), a, b, theta=theta*un.deg)
            mask = ap.to_mask()
            try:
                dat_mask[mask.bbox.iymin:mask.bbox.iymax,mask.bbox.ixmin:mask.bbox.ixmax] = mask.data
            except:
                pass
        dat_masked = np.ma.masked_array(dat, dat_mask)
        self.background.rms = rms(dat_masked)
        self.background.mean = np.nanmean(dat_masked)
        self.background.median = np.nanmedian(dat_masked)
        self.background.std = np.nanstd(dat_masked)
        return
    
    def get_source_fluxes(self, cropped=True, positions=None):
        dat, w, fn = self.get_data_and_wcs(cropped)
        if positions is None:
            poss = self.source_positions
        elif positions is not None:
            poss = positions
            
        if type(poss) is not list and poss.size ==1:
            poss = [poss]
            
        fluxes = np.zeros(len(poss))
        for i,pos in enumerate(poss):
            if pos.size != 0:
                x,y = w.world_to_pixel(pos)
                x = int(np.round(x))
                y = int(np.round(y))
                if 0 <= x < dat.shape[0] and 0 <= y < dat.shape[1]:
                    flux = dat[y,x]
                    fluxes[i] = flux
            else:
                fluxes[i] == np.nan
            
        return fluxes
    
    def plot_frame(self, source_of_interest:SkyCoord = None, cropped=True, plot_sources=True, plot_source_of_interest=False):
        dat, w, fn = self.get_data_and_wcs(cropped)
        fig, ax = plt.subplots(1,1)
        ax.imshow(dat, origin = 'lower')
        if plot_sources and self.source_positions is not None:
            for source in self.source_positions:
                x,y = w.world_to_pixel(source)
                ap = EllipticalAperture([(x,y)], self.beam.bmaj, self.beam.bmin, (90 + self.beam.bpa)*un.deg)
                ap.plot(ax=ax, color = 'r')
        if plot_source_of_interest is True and source_of_interest is not None:
            x,y = w.world_to_pixel(source_of_interest)
            ap = EllipticalAperture([(x,y)], self.beam.bmaj, self.beam.bmin, (90 + self.beam.bpa)*un.deg)
            ap.plot(ax=ax, color = 'w')
        return fig, ax
    
    def save(self, outn=None):
        if outn is None:
            outn = os.path.join(self.working_directory,'')
            outn = f"{outn}frame_class_data.npz"
        np.savez(outn, self.__dict__)
        return   
    

class observation:
    def __init__(self, source:SkyCoord, full_frame_fns, out_dir, max_sep=0.2*un.deg):
        
        self.source = source
        
        if os.path.isdir(out_dir):
            self.out_dir = out_dir
        elif not os.path.isdir(out_dir):
            raise Exception(f"{out_dir} is not an existing directory")
            
        self.full_frame_fns = full_frame_fns
        self.__init_frames(full_frame_fns, out_dir)
        
        self.max_sep = max_sep
        
        self.detected_sources = []
        self.detected_source_fluxes = np.array([])
        self.persistent_sources = None
        
        self.latest_idx_w_source = None
        self.equinox = 'fk5'
        self.n_frames = len(self.full_frame_fns)
        
        self.source_positions = np.array([]) # for the actual source of interest; not sources detected in the frames
        self.source_fluxes = np.array([])
        return
    
    def __init_frames(self, fns, out_dir):
        frames = []
        for i, fn in enumerate(fns):
            frames.append(obs_frame(fn, i, out_dir))
        self.frames = frames
        self.timestamps = [Time(frame.timestamp) for frame in self.frames]
        return
    
    def crop_frames(self, idx:int = None, dimension=100, out_subdir=None, overwrite=True):
        if idx is None:
            for frame in self.frames:
                frame.crop_frame(self.source, dimension, out_subdir, overwrite)
        elif idx is not None:
            self.frames[idx].crop_frame(self.source, dimension, out_subdir, overwrite)
        return
    
    def find_sources(self, idx:int = None, sigma_threshold=4, cropped: bool = True, verbose: bool = False):
        if idx is None:
            for i,frame in enumerate(self.frames):
                if verbose:
                    print(f"Frame {i+1}/{self.n_frames}")
                frame.get_source_positions(sigma_threshold=sigma_threshold, cropped=cropped, verbose=verbose)
        elif idx is not None:
            self.frames[idx].get_source_positions(sigma_threshold=sigma_threshold, cropped=cropped, verbose=verbose)
        return
    
    def start_reference_list(self):
        obs_sources = []
        latest_idx_w_source = 0
        while len(obs_sources) == 0:
            if self.frames[latest_idx_w_source].source_positions is not None:
                for j,s in enumerate(self.frames[0].source_positions):
                    source_name = f'source_{str(j+1).zfill(3)}'
                    positions = [SkyCoord([]*un.deg, []*un.deg, equinox=s.equinox, frame=self.equinox)]*self.n_frames
                    positions[latest_idx_w_source] = s 
                    source = radio_source(source_name, positions, s)
                    obs_sources.append(source)
            else:
                latest_idx_w_source += 1
        self.latest_idx_w_source = latest_idx_w_source
        self.detected_sources = obs_sources
        return
    
    def crossmatch(self, new_sources, max_sep: un.Quantity = None, ref_idx:int = None):
        if max_sep is not None:
            self.max_sep = max_sep
        elif max_sep is None:
            max_sep = self.max_sep
        
        ref_idx = self.latest_idx_w_source if ref_idx is None else ref_idx
        new_sources = copy.deepcopy(new_sources)
        
        most_recent_sources = SkyCoord([s.positions[ref_idx] for s in self.detected_sources if s.positions[ref_idx].size != 0]) # most recently identified sources; take priority for cross matching
        most_recent_source_names = [s.label for s in self.detected_sources if s.positions[ref_idx].size !=0] 
        
        all_sources = SkyCoord([s.avg_position for s in self.detected_sources]) # average position for all previously identified sources
        all_source_names = [s.label for s in self.detected_sources]
        
        assigned_sources = {} # keeps track of which source labels have been assigned
        
        idx_recent, seps_recent, __ = new_sources.match_to_catalog_sky(most_recent_sources)
        idx_recent_uniq = np.unique(idx_recent)
        
        # get the sources that were actually near a source in the previous frame
        for idx in idx_recent_uniq:
            ref_source = most_recent_sources[idx]
            new_source_idx, new_source_sep, __ = ref_source.match_to_catalog_sky(new_sources) # the index of the new source closest to the source in the previous frame
            if new_source_sep <= max_sep: # make sure the closest source is actually within the distance cut off
                assigned_sources.update({f'{most_recent_source_names[idx]}': new_sources[new_source_idx]})
                new_sources = new_sources[[i for i in range(len(new_sources)) if i != new_source_idx]] # removes the source so it doesn't get double assigned
                
        # go through the rest of the sources that didn't get an assignment to a source in the previous frame to see if they're near any previously-identified source
        idx_all, seps_all, __ = new_sources.match_to_catalog_sky(all_sources)
        idx_all_uniq = np.unique(idx_all)
        for idx in idx_all_uniq:
            ref_source = all_sources[idx]
            new_source_idx, new_source_sep, __ = ref_source.match_to_catalog_sky(new_sources)
            if new_source_sep <= max_sep:
                assigned_sources.update({f'{all_source_names[idx]}': new_sources[new_source_idx]})
                new_sources = new_sources[[i for i in range(len(new_sources)) if i != new_source_idx]] 
        
        # any sources not matched to a previously-identified source is a new source
        latest_source_number = int(all_source_names[-1].split('_')[-1])
        count = 1
        for ns in new_sources:
            assigned_sources.update({f"source_{str(latest_source_number + count).zfill(3)}":ns})
        return assigned_sources
    
    def build_observation_source_list(self, max_sep=None):
        if max_sep is None:
            max_sep = self.max_sep
            
        for i, frame in enumerate(self.frames):
            assigned_sources = self.crossmatch(frame.source_positions, max_sep)
            
            if len(assigned_sources) != 0:
                source_list = [s.label for s in self.detected_sources]
                self.latest_idx_w_source = i
                
                for key in list(assigned_sources.keys()):
                    if key in source_list:
                        source_idx = source_list.index(key)
                        count = self.detected_sources[source_idx].frame_count
                        self.detected_sources[source_idx].positions[i] = assigned_sources[key]
                        avg_orig = self.detected_sources[source_idx].avg_position
                        ra_weight, dec_weight = (avg_orig.ra*count, avg_orig.dec*count)   
                        ra_new = (ra_weight + assigned_sources[key].ra)/(count+1)
                        dec_new = (dec_weight + assigned_sources[key].dec)/(count+1)
                        # sc = SkyCoord(self.detected_sources[source_idx][:i+1].positions) # too slow
                        self.detected_sources[source_idx].avg_position = SkyCoord(ra_new, dec_new, equinox=assigned_sources[key].equinox, frame=self.equinox)
                        self.detected_sources[source_idx].frame_count += 1
                        
                    elif key not in source_list:
                        positions = [SkyCoord([]*un.deg, []*un.deg, equinox=assigned_sources[key].equinox, frame=self.equinox)]*self.n_frames
                        positions[i] = assigned_sources[key]
                        source = radio_source(key, positions, assigned_sources[key])
                        source.frame_count += 1
                        self.detected_sources.append(source)
        return
    
    def calc_space_changes(self):
        for i, source in enumerate(self.detected_sources):
            self.detected_sources[i].calc_space_change()
        return
    
    def get_fluxes_single_frame(self, frame_idx:int, get_bkg=False, cropped=True):
        frame = self.frames[frame_idx]
        dat, w, fn = frame.get_data_and_wcs(cropped)
        if get_bkg:
            frame.get_background_stats(cropped)
        
        positions = [s.positions[frame_idx] for s in self.detected_sources]
        fluxes = frame.get_source_fluxes(cropped, positions)
        
        return fluxes
    
    def assign_persistent_sources(self, n_sources: int = None, frame_fraction=None):
        frame_counts = [s.frame_count for s in self.detected_sources]
        sources_sorted = [self.detected_sources[i] for i in np.argsort(frame_counts)]
        sources_sorted.reverse()
        frame_counts.sort(reverse=True)
        
        if n_sources is None:
            if frame_fraction is None:
                mean_detect = np.nanmean(np.array(frame_counts))
                persistent_sources = np.array(sources_sorted)[np.array(frame_counts) >= mean_detect]
            elif frame_fraction is not None:
                persistent_sources = np.array(sources_sorted)[np.array(frame_counts) >= self.n_frames*frame_fraction]
                
        elif n_sources is not None:
            persistent_sources = sources_sorted[:n_sources]
        
        fluxes = [np.nanmedian(s.fluxes) for s in persistent_sources]
        persistent_sources = [persistent_sources[i] for i in np.argsort(fluxes)]
        persistent_sources.reverse()
        self.persistent_sources = persistent_sources
        return
    
    def get_fluxes_all_frames(self, get_bkg=False, cropped=True):
        """
        Gets fluxes for all sources across all frames
        """
        fluxes = np.zeros((self.n_frames, len(self.detected_sources)))
        for i, frame in enumerate(self.frames):
            fluxes[i] = self.get_fluxes_single_frame(i, get_bkg, cropped)
            
        for i, s in enumerate(self.detected_sources):
            s.fluxes = fluxes.transpose()[i]
        self.detected_source_fluxes = fluxes    
        return 
    
    def find_weighted_position_change(self, frame_idx:int = None):
        pi2 = np.pi/2 * un.rad
        source_star_separations = []
        position_vectors = []
        
        if self.persistent_sources is None:
            self.assign_persistent_sources()
            
        for source in self.persistent_sources:
            if source.seps is None:
                source.calc_space_change()
            if source.positions[frame_idx].size != 0:    
                mag = source.seps[frame_idx].to('arcmin').value
                ang = -source.pos_angs[frame_idx] + 5*pi2
                
                position_vectors.append([mag*np.cos(ang), mag*np.sin(ang)])
                source_star_separations.append(self.source.separation(source.avg_position).to('arcmin').value)
                
        if len(source_star_separations) != 0:
            source_star_separations = np.array(source_star_separations)
            weights = source_star_separations.sum()/source_star_separations
            vecs_sum = np.average(position_vectors, axis = 0, weights=weights)
            sep = (vecs_sum**2).sum()**0.5
            theta = np.arctan(vecs_sum[1]/vecs_sum[0])*un.rad
            if vecs_sum[0] < 0:
                theta += np.pi*un.rad
            # if theta < 0:
            #     theta += 2*np.pi*un.rad
            theta = -theta + 5*pi2
        elif len(source_star_separations) == 0:
            warnings.warn(f"No sources found in the field to estimate position offset for frame {frame_idx}")
            sep = 0
            theta = 0*un.rad
        return sep*un.arcmin, theta
    
    def get_star_position(self, frame_idx):
        """
        Gets the new position of the star/coordinate based on weighted position change found for a given frame
        :param frame_key: the key for the frame in frame_source_dict to get the new star/coordinate position
        :type frame_key: str
        :return fixed_coord: new coordinate of the star/coordinate
        :rtype fixed_coord: astropy.coordinates.SkyCoord
        """
        sep, theta = self.find_weighted_position_change(frame_idx=frame_idx)
        fixed_coord = self.source.directional_offset_by(theta, sep)
        return fixed_coord
    
    def get_all_star_positions(self):
        """
        Gets the position of the star for all frames and update self.source_positions.
        """
        positions = []
        for i in range(self.n_frames):
            fixed_coord = self.get_star_position(i)
            positions.append(fixed_coord)
        self.source_positions = SkyCoord(positions)
        return
    
    def get_star_fluxes(self, cropped=True):
        if self.source_positions is None:
            print("Finding corrected star/coordinate positions first")
            self.get_all_star_positions()
            
        pix_flux = np.zeros(self.n_frames)
        for i in range(self.n_frames):
            dat, w, fn = self.frames[i].get_data_and_wcs(cropped)
            
            x,y = w.world_to_pixel(self.source_positions[i])
            x = int(np.round(x))
            y = int(np.round(y))
            pix_flux[i] = dat[y,x]
        self.source_fluxes = pix_flux
        return
    
    def save(self, outn=None):
        if outn is None:
            outn = os.path.join(self.out_dir,'')
            outn = f"{outn}observation_class_data.npz"
        np.savez(outn, self.__dict__)
        return
    

class radio_source:
    def __init__(self, label:str, positions = None, avg_position=None, seps = None, pos_angs=None, fluxes=None):
        self.label = label
        self.positions = positions
        self.seps = seps
        self.pos_angs = pos_angs
        self.fluxes = fluxes
        self.avg_position = avg_position
        self.frame_count = 0
        return
    
    def calc_space_change(self, return_vals=False):
        pos_angles = np.zeros(len(self.positions))
        separations = np.zeros(len(self.positions))
        for j, _ in enumerate(pos_angles):
            if self.positions[j].size != 0:
                pos_angles[j] = self.avg_position.position_angle(self.positions[j]).to('deg').value
                separations[j] = self.avg_position.separation(self.positions[j]).to('arcmin').value
            else:
                separations[j] = np.nan
                pos_angles[j] = np.nan
            
        self.pos_angs = pos_angles * un.deg
        self.seps = separations * un.arcmin
        if return_vals:
            return separations, pos_angles
        return
    
    def save(self, outn):
        np.savez(outn, self.__dict__)
        return
        
    
        
def load_observation(fp):
    """
    loads in a lightcurve_extraction
    :param fp: the filepath and name of the file resulting from lightcurve_extraction.save_data
    :type fp: str
    """
    d = np.load(fp, allow_pickle=True)['arr_0'].flatten()[0]
    obs = observation(source=d['source'], full_frame_fns=d['full_frame_fns'], out_dir=d['out_dir'])
    for k in d.keys():
        obs.__dict__[k] = d[k]
    return obs

def load_source(fp):
   d = np.load(fp, allow_pickle=True)['arr_0'].flatten()[0]
   source = radio_source(label=d['label'], positions=d['positions'], avg_position=d['avg_position'], seps=d['seps'], pos_angs=d['pos_angs'], fluxes=d['fluxes'])
   for k in d.keys():
       source.__dict__[k] = d[k]
   return source

def load_frame(fp):
    d = np.load(fp, allow_pickle=True)['arr_0'].flatten()[0]
    frame = radio_source(file_name=d['file_name'], index=d['index'], working_directory=d['working_directory'])
    for k in d.keys():
        frame.__dict__[k] = d[k]
    return frame
  
            