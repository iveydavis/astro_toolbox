import numpy as np
import scipy.optimize as opt

from astropy.io import fits
from astropy import units as un
from astropy.time import Time
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from photutils import DAOStarFinder, EllipticalAperture
from photutils.detection import find_peaks

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


def get_beam_shape_params_iter(fn, update_file_header=True, subframe_radius = (20,20), std_threshold = 6, niter = 21, sma=15, eps=0.7, pa=2):
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
    
    w = WCS(f.header, naxis = 2)
    w._naxis = [w._naxis[0], w._naxis[1]]
    
    pks = find_peaks(data, threshold = std_threshold * rms(data) + np.nanmedian(data))
    pks.sort('peak_value', reverse=True)
    
    thresh = 5*std_threshold # preliminary threshold to enter the while loop
    data_std = np.nanstd(data)
    
    for i, pk in enumerate(pks):
        subframe = data_new[pk[1]-subframe_radius[0]:pk[1]+subframe_radius[0], pk[0]-subframe_radius[1]:pk[0]+subframe_radius[1]]
        thresh = np.nanstd(subframe)/data_std
        if thresh < std_threshold:
            break
        if i == niter:
            break
    
    if thresh >= std_threshold:
        # if it never reaches a std that suggests there's only one source in the subframe to extract a solution from
        warnings.warn(f"STD threshold was never met ({thresh})")
        popt = np.array([0,0,0,0,0,0,0])
    
    elif thresh < std_threshold:
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
        """
        Descriptor for the beam shape in an image
        :param bmaj: Major axis of the beam in pixels; BMAJ/CDELT2 quantities in the header
        :type bmaj: "pix"
        :param bmin: Minor axis of the beam in pixels; BMIN/CDELT2 quantities in the header
        :type bmin: "pix"
        :param bpa: Beam position angle in degrees; BPA in the header
        :type bpa: "deg"
        :return: beam instance
        :rtype: beam

        """
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        return
        
class background:
    def __init__(self, mean, median, std, rms):
        """
        Statistics for an image/ data in an obs_frame class
        :param mean: mean value
        :type mean: float
        :param median: median value in the image
        :type median: float
        :param std: standard deviation in the image
        :type std: float
        :param rms: root-mean square of an image; calculated from the rms function
        :type rms: float
        :return: background instance
        :rtype: background

        """
        self.mean = mean
        self.median = median
        self.std = std
        self.rms = rms
        return
    
class obs_frame:
    def __init__(self, file_name, index=None, working_directory=None, cropped=False):
        """
        Class for holding information for a frame in an observation
        :param file_name: The name of the FITS file that has the data
        :type file_name: str
        :param index: The index of the frame within the larger observation. Not necessary if only investigating one frame, defaults to None
        :type index: int, optional
        :param working_directory: The directory that cropped frames will get saved to. If None, it defaults to the current working directory, defaults to None
        :type working_directory: str, optional
        :return: obs_frame instance
        :rtype: obs_frame

        """
        self.source_positions = None
        self.file_name = file_name
        self.timestamp = None
        self.__update_frame()
        self.__get_working_directory(working_directory)
        self.__get_beam()
        if not cropped:
            self.cropped_filepath= None
            self.cropped_wcs = None
        elif cropped:
            self.cropped_filepath = file_name
            self.cropped_wcs = self.wcs
        self.index = index
        self.background = background(None, None, None, None)
        return
    
    def __update_frame(self):
        """
        updates the obs_frame instance with the header, WCS, and time of the observation from the FITS header

        """
        f = fits.open(self.file_name)
        self.header = f[0].header
        self.wcs = WCS(self.header, naxis=2)
        self.wcs._naxis = [self.wcs._naxis[0],self. wcs._naxis[1]]
        self.timestamp = self.header['DATE-OBS']
        f.close()
        return 
    
    def __get_working_directory(self, working_directory):
        """
        Sets the working directory path
        """
        if working_directory is not None:
            self.working_directory = working_directory
        elif working_directory is None:
            self.working_directory = os.getcwd()
            warnings.warn(f"Working directory not specified, taking it to be {self.working_directory}")
        return
    
    def __get_beam(self):
        """
        Gets the beam information from the header. If the beam information isn't there, then it tries to derive the beam information from sources in the frame
        """
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
    
    def crop_frame(self, center:SkyCoord, dimension: "pix" = 100, out_subdir=None, overwrite: bool = True):
        """
        Crops a frame to a small region centered on a source of interest. Updates the cropped_filepath and cropped_wcs properties
        :param center: The SkyCoord position where the image should be centered on
        :type center: SkyCoord
        :param dimension: The x,y dimension of the cropped frame, defaults to 100
        :type dimension: integer number of pixels, optional
        :param out_subdir: The sub-directory under self.working_directory that cropped frames will get saved to. If None, data get saved to a subdirectory called cropped_frames/, defaults to None
        :type out_subdir: str, optional
        :param overwrite: If True, overwrites files of the same name that may already exist in the sub_dir, defaults to True
        :type overwrite: bool, optional
        """
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
        
        header_orig_keys = list(self.header.keys())
        header_new = sub_frame.wcs.to_header()
        header_new_keys = list(header_new.keys())
        for hk in header_orig_keys:
            if hk not in header_new_keys and hk != "COMMENT" and hk!= "HISTORY":
                header_new.append((hk, self.header[hk]))
        
        fits.writeto(out_name, sub_frame.data, header_new, overwrite=overwrite)
        self.cropped_filepath = out_name
        self.cropped_wcs = sub_frame.wcs
        return
    
    def get_data_and_wcs(self, cropped: bool):
        """
        Returns the data, WCS, and fn for a file based on whether the cropped version is requested or not
        :param cropped: Indicates whether the returned data should be the cropped or original data
        :type cropped: bool
        :return: Data, WCS, and filename of the frame
        :rtype: numpy.ndarray, astropy.wcs.WCS, str

        """
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
    
    def get_source_positions(self, sigma_threshold=4, cropped: bool = True, verbose: bool = False, do_max_pix_cut=False):
        """
        Gets the positions of sources in the frame and saves the positions in the source_positions property
        :param sigma_threshold: The factor of std that a source needs to be above the median value to be included as a source, defaults to 4
        :type sigma_threshold: float, optional
        :param cropped: If True, it searches for sources in the cropped frame. If False, it looks for sources in the original frame, defaults to True
        :type cropped: bool, optional
        :param verbose: If true, it will print the number of sources found, defaults to False
        :type verbose: bool, optional
        """
        dat, w, fn = self.get_data_and_wcs(cropped)
            
        dao = DAOStarFinder(threshold=sigma_threshold*rms(dat) + np.nanmedian(dat), fwhm=self.beam.bmaj*2, ratio=self.beam.bmin/self.beam.bmaj, theta=(90 + self.beam.bpa))
        sources = dao(dat)
        if len(sources) != 0:
            if do_max_pix_cut:
                sources_clip = sources[sources['peak'] > sigma_threshold*rms(dat) + np.nanmedian(dat)]
            else:
                sources_clip = sources
            source_clip_sc = w.pixel_to_world(sources_clip['xcentroid'], sources_clip['ycentroid'])
            self.source_positions = source_clip_sc
        elif len(sources) == 0:
            self.source_positions = SkyCoord([]*un.deg, []*un.deg)
        if verbose:
            print(f"{len(sources)} found in {fn}")
        return
    
    def get_background_stats(self, cropped: bool = True):
        """
        Gets the background statistics by masking sources indicated by source_positions property. Statistics are saved to the background property
        :param cropped: If cropped, it calculates statistics for the cropped frame, defaults to True
        :type cropped: bool, optional

        """
        dat, w, fn = self.get_data_and_wcs(cropped)
        dat_mask = np.zeros(dat.shape)
        
        a = self.beam.bmaj
        b = self.beam.bmin
        theta = self.beam.bpa + 90
        
        sources = self.source_positions
        if sources is not None:
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
    
    def get_source_fluxes(self, cropped: bool = True, positions=None):
        """
        Gets the fluxes of the sources
        :param cropped: Indicates whether cropped files should be used, defaults to True
        :type cropped: bool, optional
        :param positions: Specific source positions that can be used to extract fluxes from. If None, then it uses the source positions in the source_position property, defaults to None
        :type positions: list, optional
        :return: The fluxes for the sources in the order of the list
        :rtype: numpy.ndarray

        """
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
    
    def plot_frame(self, source_of_interest:SkyCoord = None, cropped: bool = True, plot_sources: bool = True, plot_source_of_interest: bool = False):
        """
        Plots the frame in the wcs projection and optionally the sources that were identified
        :param source_of_interest: The position of the source of interest, defaults to None
        :type source_of_interest: SkyCoord, optional
        :param cropped: Indicates whether it should plot the cropped frame (True) or original frame (False), defaults to True
        :type cropped: bool, optional
        :param plot_sources: If True, it will plot ellipses at the position of the sources, defaults to True
        :type plot_sources: bool, optional
        :param plot_source_of_interest: If True, it will plot an ellipse at the position of the source of interest, defaults to False
        :type plot_source_of_interest: bool, optional
        :return: The figure and axis instance of the plot
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        """
        dat, w, fn = self.get_data_and_wcs(cropped)
        fig = plt.figure()
        ax = fig.add_subplot(projection=w)
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
        """
        Saves the instance information that can be loaded in later
        :param outn: The filepath for the data to be saved to. If None, it saves it in the working_directory as 'frame_class_data.npz', defaults to None
        :type outn: str, optional

        """
        if outn is None:
            outn = os.path.join(self.working_directory,'')
            outn = f"{outn}frame_class_data.npz"
        np.savez(outn, self.__dict__)
        return   
    

class observation:
    def __init__(self, source:SkyCoord, full_frame_fns, out_dir: str, max_sep=0.2*un.deg, freq_range=(), cropped: bool = False, load: bool = False):
        """
        Information on sources in a full observation of a given subband. Used for estimating the ionosphere's impact on source positions so that the flux can be derived from the correct location
        :param source: The position of the source of interest. Images will be cropped around this location
        :type source: SkyCoord
        :param full_frame_fns: a list of FITS file names for the data
        :type full_frame_fns: list of str
        :param out_dir: The directory that cropped data will be saved to. Must already exist
        :type out_dir: str
        :param max_sep: the maximum distance that a source is allowed to be from its other positions to still be considered associated with a source label, defaults to 0.2*un.deg
        :type max_sep: astropy.units.quantity.Quantity, optional
        :raises Exception: if the out_dir is not an existing directory
        """
        self.source = source
        
        if os.path.isdir(out_dir):
            self.out_dir = out_dir
        elif not os.path.isdir(out_dir):
            raise Exception(f"{out_dir} is not an existing directory")
        
        
        self.full_frame_fns = full_frame_fns
        if not load:
            self.__init_frames(full_frame_fns, out_dir, cropped)
        
        self.max_sep = max_sep
        
        self.detected_sources = []
        self.persistent_sources = None
        
        self.latest_idx_w_source = None
        self.equinox = 'fk5'
        self.n_frames = len(self.full_frame_fns)
        
        self.source_positions = np.array([]) # for the actual source of interest; not sources detected in the frames
        self.source_fluxes = np.array([])
        
        self.freq_range = self.__get_freq_range(freq_range)
        return
    
    
    def __init_frames(self, fns, out_dir, cropped):
        """
        Initializes the obs_frame instances and assigns their indices
        """
        frames = []
        for fn in fns:
            frames.append(obs_frame(fn, working_directory=out_dir, cropped=cropped))
            
        frames = frames
        timestamps = [Time(frame.timestamp) for frame in frames]
        ts_mjd = [ts.mjd for ts in timestamps]
        
        frames_sorted = [f for _,f in sorted(zip(ts_mjd, frames))]
        for i, frame in enumerate(frames_sorted):
            frame.index = i
        self.frames = frames_sorted
        
        ts_mjd.sort()
        self.timestamps = Time(ts_mjd, format='mjd')
        return
    
    def __get_freq_range(self, freq_range):
        if len(freq_range) == 0:
            try:
                hdr = fits.open(self.full_frame_fns[0])[0].header
                center = hdr['CRVAL3']
                halfband = hdr['CDELT3']/2
                freq_range = (center - halfband, center + halfband)
            except:
                raise Warning(f"Could not get frequency range from {self.full_frame_fns[0]}")
                freq_range = ()
        return freq_range
    
    def crop_frames(self, idx:int = None, dimension: "pix" = 100, out_subdir=None, overwrite: bool = True):
        """
        Crops all of the obs_frame instances in the frames property to the same size
        :param idx: If only wanting to crop one frame, can specify the index of the frame to crop. If None, it crops all frames, defaults to None
        :type idx: int, optional
        :param dimension: The x,y dimension to crop the frames to, defaults to 100
        :type dimension: integer number of pixels, optional
        :param out_subdir: The subdirectory under working_directory to save data to. If None, data get saved to a subdirectory called cropped_frames/ defaults to None
        :type out_subdir: str, optional
        :param overwrite: If True, overwrites files of the same name that may already exist in the sub_dir, defaults to True
        :type overwrite: bool, optional
        """
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
        """
        Makes a starting list of sources that will be used as the starting crossmatch for future sources.
        Updates the latest_idx_w_source propert, which keeps track of the index of the index of the frame that most recently had detected sources used to make the reference list
        Updates detected_sources with the sources that were detected in the frame. This is the reference list

        """
        obs_sources = []
        latest_idx_w_source = 0
        while len(obs_sources) == 0:
            if self.frames[latest_idx_w_source].source_positions is not None:
                for j,s in enumerate(self.frames[0].source_positions):
                    source_name = f'source_{str(j+1).zfill(3)}'
                    positions = [SkyCoord([]*un.deg, []*un.deg, equinox=s.equinox, frame=self.equinox)]*self.n_frames
                    positions[latest_idx_w_source] = s 
                    source = radio_source(source_name, positions, s, timestamps=self.timestamps, freq_range=self.freq_range)
                    obs_sources.append(source)
            else:
                latest_idx_w_source += 1
        self.latest_idx_w_source = latest_idx_w_source
        self.detected_sources = obs_sources
        return
    
    def crossmatch(self, new_sources, max_sep: un.Quantity = None, ref_idx:int = None):
        """
        Cross matches new_sources with the positions of self.detected_sources to figure out the association between sources of different frames
        :param new_sources: the list of source positions to be cross matched with the sources in the detected_sources property
        :type new_sources: list of SkyCoord
        :param max_sep: the maximum allowed separation between sources in two different frames for them to be considered the same. If None, it uses the max_sep property as the quantity, defaults to None
        :type max_sep: un.Quantity, optional
        :param ref_idx: The frame index to reference against other. If None, it uses the latest_idx_w_source property, defaults to None
        :type ref_idx: int, optional
        :return: A dictionary of sources and their association label
        :rtype: dict

        """
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
        """
        Crossmatches sources among all of the frames and builds the full list of sources that show up across the full observation
        :param max_sep: The maximum separation allowed for a source in two different frames to be considered associated, If None, it uses the value in the max_sep property of the class. defaults to None
        :type max_sep: astropy.units.quantity.Quantity, optional

        """
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
                        source = radio_source(key, positions, assigned_sources[key], timestamps=self.timestamps, freq_range=self.freq_range)
                        source.frame_count += 1
                        self.detected_sources.append(source)
        return
    
    def calc_space_changes(self):
        """
        Calculates the separation and position angle difference between a source in a given frame and its average position. Does this for all detected sources

        """
        for i, source in enumerate(self.detected_sources):
            self.detected_sources[i].calc_space_change()
        return
    
    def get_fluxes_single_frame(self, frame_idx:int, get_bkg: bool = False, cropped: bool = True):
        """
        Gets the fluxes for sources in a single frame
        :param frame_idx: The index of the frame to get source fluxes from
        :type frame_idx: int
        :param get_bkg: If True, it will also get the background statistics, defaults to False
        :type get_bkg: bool, optional
        :param cropped: Indicates whether to use the cropped data, defaults to True
        :type cropped: bool, optional
        :return: The fluxes for the sources in the order of the sources listed in the frame
        :rtype: np.ndarray

        """
        frame = self.frames[frame_idx]
        dat, w, fn = frame.get_data_and_wcs(cropped)
        if get_bkg:
            frame.get_background_stats(cropped)
        
        positions = [s.positions[frame_idx] for s in self.detected_sources]
        fluxes = frame.get_source_fluxes(cropped, positions)
        
        return fluxes
    
    def assign_persistent_sources(self, n_sources: int = None, frame_fraction=None):
        """
        Identifies the most persistently occuring sources. Updates persistent_sources property to be used for estimating source motion
        :param n_sources: The number of sources to be considered persistent. If None then the number of persistent sources is determined by either the fraction of frames the source shows up in or the mean number of frames that sources show up in, defaults to None
        :type n_sources: int, optional
        :param frame_fraction: If this is not None and n_sources is None, then it is used to estimate the number of persistent sources based on the fraction of frames (<1) a source shows up in, defaults to None
        :type frame_fraction: float < 1, optional

        """
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
    
    def get_fluxes_all_frames(self, get_bkg: bool = True, cropped: bool = True):
        """
        Gets fluxes for all sources across all frames and updates the detected_source_fluxes property
        :param get_bkg: If True, it gets the background statistics for the frames, defaults to False
        :type get_bkg: bool, optional
        :param cropped: Indicates if the data to be used is cropped, defaults to True
        :type cropped: bool, optional

        """
        fluxes = np.zeros((self.n_frames, len(self.detected_sources)))
        for i, frame in enumerate(self.frames):
            fluxes[i] = self.get_fluxes_single_frame(i, get_bkg, cropped)
            
        for i, s in enumerate(self.detected_sources):
            s.fluxes = fluxes.transpose()[i] 
        return 
    
    def find_weighted_position_change(self, frame_idx:int = None):
        """
        Uses the persistent sources' position changes to estimate the position of the source of interest
        :param frame_idx: The index of the frame to get the position change. If None, it does it for all frames, defaults to None
        :type frame_idx: int, optional
        :return: The separation and position angle change of the source of interest
        :rtype: astropy.units.quantity.Quantity, astropy.units.quantity.Quantity

        """
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
    
    def get_star_fluxes(self, cropped: bool = True):
        """
        Gets the flux of the source of interest and saves the information in self.source_fluxes
        :param cropped: Indicates if the data to be used is cropped, defaults to True
        :type cropped: bool, optional

        """
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
    
    def process(self, sigma_threshold=4, cropped: bool = True, verbose: bool = False, max_sep=0.2*un.deg, n_persistent=4):
        """
        Does all of the steps to estimate the position of the source of interest and extract its flux
        :param sigma_threshold: The factor of std that a source needs to be above the median value to be included as a source, defaults to 4
        :type sigma_threshold: float, optional
        :param cropped: If True, it searches for sources in the cropped frame. If False, it looks for sources in the original frame, defaults to True
        :type cropped: bool, optional
        :param verbose: If true, it will print the number of sources found, defaults to False
        :type verbose: bool, optional
        :param max_sep: Maximum allowed separation for sources found in two frames to be considered the same radio_source, defaults to 0.2*un.deg
        :type max_sep: astropy.units.quantity.Quantity, optional
        """
        self.find_sources(sigma_threshold=sigma_threshold, cropped=cropped, verbose=verbose)
        self.start_reference_list()
        self.build_observation_source_list(max_sep)
        self.get_fluxes_all_frames()
        self.assign_persistent_sources(n_persistent)
        self.get_all_star_positions()
        self.get_star_fluxes()
        return
    

class radio_source:
    def __init__(self, label:str, positions = None, avg_position=None, seps = None, pos_angs=None, fluxes=None, timestamps: Time = None, freq_range=None):
        """
        Holds information for unique sources identified in an observation
        :param label: The label for the source
        :type label: str
        :param positions: The positions of the source throughout an observation, defaults to None
        :type positions: list of SkyCoord, optional
        :param avg_position: The average position of the source across the observation, defaults to None
        :type avg_position: SkyCoord, optional
        :param seps: The separation of the source in each frame from its average position, defaults to None
        :type seps: numpy.ndarray, optional
        :param pos_angs: The position angles of the source in each frame relative to its average position, defaults to None
        :type pos_angs: numpy.ndarray, optional
        :param fluxes: The flux of the source in each frame, defaults to None
        :type fluxes: numpy.ndarray, optional

        """
        self.label = label
        self.positions = positions
        self.seps = seps
        self.pos_angs = pos_angs
        self.fluxes = fluxes
        self.avg_position = avg_position
        self.frame_count = 0
        self.timestamps = timestamps
        self.freq_range = freq_range
        return
    
    def calc_space_change(self, return_vals=False):
        """
        Calculates the separations and position angles to be stored in the seps and pos_angs properties respectively
        :param return_vals: If True, it will return the separations and position angles in addition to updating the seps and pos_angs properties, defaults to False
        :type return_vals: bool, optional
        :return: if return_vals is True, the return values are the separations and position angles of the source relative to its average position
        :rtype: numpy.ndarray, numpy.ndarray

        """
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
    
    def plot_light_curve(self, xaxis:str = 'frame'):
        """
        Plots the light curve for the source
        :param xaxis: Describes whether the xaxis is in frames or in mjd. Options are 'frame' and 'mjd', defaults to 'frame'
        :type xaxis: str, optional
        :return: the figure and axis instance of the plot
        :rtype: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        """
        allowed_xaxis = ['frame', 'mjd']
        assert(xaxis.lower() in allowed_xaxis), f"xaxis must be in {allowed_xaxis}"
        
        if xaxis.lower() == 'frame':
            xvec = np.linspace(0,len(self.fluxes)-1, len(self.fluxes))
            xlabel = 'Frame number'
            
        elif xaxis.lower() == 'mjd':
            xvec = self.timestamps.mjd
            xlabel = 'MJD'
            
        fig, ax = plt.subplots(1,1)
        ax.plot(xvec, self.fluxes)
        ax.set_ylabel('Flux density', fontsize = 14)
        ax.set_xlabel(xlabel, fontsize = 14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        return fig, ax
    
    def save(self, outn: str):
        """
        Saves the data to outn
        :param outn: The file name including path to save the data to
        :type outn: str

        """
        np.savez(outn, self.__dict__)
        return
    

        
class multi_freq_source:
    def __init__(self, label, sources):
        """
        
        :param label: the label of the source identified across multiple frequency bands
        :type label: str
        :param sources: the radio_source objects associated with this broad_spectrum source
        :type sources: list of radio_source objects
        """
        self.label = label
        self.sources = sources
        self.center_freqs = [np.mean(source.freq_range) for source in sources]
        self.df = [np.diff(source.freq_range)[0] for source in sources]
        self.dynamic_spectrum = None
        self.__sort_by_frequency()
        return
    
    def __sort_by_frequency(self):
        df_sorted = [df for _,df in sorted(zip(self.center_freqs,self.df))]
        self.center_freqs.sort()
        self.df = df_sorted
        return
    
    def standardize_timestamps(self):
        all_timestamps = []
        for source in list(self.sources):
            all_timestamps = list(set(all_timestamps + source.timestamps.tolist()))
        all_timestamps.sort()
        self.all_timestamps = all_timestamps
        return
    
    def make_dynamic_spectrum(self, do_return=True):
        dyn_spec = np.zeros((len(self.all_timestamps), len(self.sources))) * np.nan
        for f, source in enumerate(self.sources):
            for i, flux in enumerate(source.fluxes):
                assert(len(source.fluxes) == len(source.timestamps))
                idx = self.all_timestamps.index(source.timestamps[i])
                dyn_spec[idx, f] = flux
                    
        self.dynamic_spectrum = dyn_spec.transpose()
        if do_return:
            return self.dynamic_spectrum
    
    def plot_dynspec(self, vmin = 0, vmax = 50, cmap='viridis', time_axis='hr'):
        """
        Plots the dynmaic spectrum
        :param vmin: minimum flux density value for the waterfall plot, defaults to 0
        :type vmin: float, optional
        :param vmax: maximum flux density value for the waterfall plot, defaults to 50
        :type vmax: float, optional

        """
        if time_axis.lower() == 'hr':
            tv = [t.mjd for t in self.timestamps - self.timestamps[0].mjd]
            tv = np.array(tv)*24
            xaxis_label = "Time since start [hr]"
        elif time_axis.low() != 'hr':
            tv = [t.mjd for t in self.timestamps - self.timestamps[0].mjd]
            tv = np.array(tv)
            xaxis_label = "Time since start [day]"
        freq_min = self.center_freqs[0] - self.df[0]/2
        freq_max = self.center_freqs[-1] + self.df[0]/2
        
        extents = [tv[0], tv[-1], freq_min/1e6, freq_max/1e6]
        
        fig, ax_dynspec = plt.subplots()
        im = ax_dynspec.imshow(self.dynamic_spectrum, aspect = 'auto',origin = 'lower',extent=extents, cmap=cmap, interpolation='nearest')
        im.set_clim(vmin=vmin, vmax=vmax)
        
        cbar = fig.colorbar(im, orientation='vertical')
        cbar.set_label(r"Flux density [Jy]", fontsize=14)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=12)
        
        ax_dynspec.set_ylabel('Frequency [MHZ]', fontsize=15)
        ax_dynspec.tick_params(axis='both', which='major', labelsize=14)
        ax_dynspec.tick_params(axis='y', which='both', labelsize=14)
        ax_dynspec.set_xlabel(xaxis_label, fontsize=15)
        return
    


class multi_freq_obs:
    def __init__(self, observations):
        self.observations = observations
        self.frequencies = np.array([np.mean(o.freq_range)] for o in observations)
        self.multi_freq_sources = []
        self.multi_freq_star = None
        self.__standardize_timestamps()
        self.__sort_by_frequency()
        return
    
    def __sort_by_frequency(self):
        obs_sorted = [obs for _,obs in sorted(zip(self.frequencies,self.observations))]
        self.observations = obs_sorted
        self.frequencies.sort()
        return
    
    def __standardize_timestamps(self):
        all_timestamps = []
        for source in list(self.sources):
            all_timestamps = list(set(all_timestamps + source.timestamps.tolist()))
        all_timestamps.sort()
        self.all_timestamps = all_timestamps
        return
    
    def process_observations(self, sigma_threshold=4, cropped: bool = True, verbose: bool = False,  max_sep=0.2*un.deg, n_persistent=4):
        """
        Processes all of the observations
        :param sigma_threshold: DESCRIPTION, defaults to 4
        :type sigma_threshold: TYPE, optional
        :param cropped: DESCRIPTION, defaults to True
        :type cropped: bool, optional
        :param verbose: DESCRIPTION, defaults to False
        :type verbose: bool, optional
        :param max_sep: DESCRIPTION, defaults to 0.2*un.deg
        :type max_sep: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        for o in self.observations:
            o.process(sigma_threshold, cropped, verbose, max_sep, n_persistent)
        return
    

    
    def find_common_sources(self, max_sep = 0.1*un.deg):
        """
        Finds sources that are associated with each other across multiple bands
        :param cross_match: maximum allowable separation for sources in two bands to be associated with each other, defaults to 0.1*un.deg
        :type cross_match: astropy.units.quantity.Quantity, optional

        """
        ref_sources = self.observations[0].persistent_sources
        
        for j, ref_source in enumerate(ref_sources):
            source_list = [ref_source]
            
            for i in range(len(self.observations)-1):
                sources = self.observations[i + 1].persistent_sources
                positions = [source.avg_position for source in sources]
                idx, sep, __  = ref_source.avg_position.match_to_catalog_sky(SkyCoord(positions))
                
                if sep < max_sep:
                    source_list.append(sources[idx])
                    
            mf_source = multi_freq_source(j, source_list)
            self.multi_freq_sources.append(mf_source)
                    
        return
    
    def make_multi_freq_star(self):
        """
        Makes a broad-band source instance for the source/star of interest

        """
        sources = []
        for obs in self.observations:
            source_positions = obs.source_positions
            fluxes = obs.source_fluxes
            source = radio_source('source_of_interest', source_positions, obs.source, fluxes=fluxes, timestamps=obs.timestamps, freq_range=obs.freq_range)
            sources.append(source)
        self.multi_freq_star = multi_freq_source('source_of_interest', sources)
        return
    
    
def process_multi_freq_obs(source_loc:SkyCoord, out_dir=None, frame_dirs = [], freq_ranges=[]):
    import glob
    assert(len(frame_dirs) == len(freq_ranges))
    observations = []
    for i, d in enumerate(frame_dirs):
        fns = glob.glob(d)
        fns.sort()
        print(f"Building observation for images in {d}")
        o = observation(source_loc, fns, out_dir, freq_range=freq_ranges[i])
        o.process()
        
    mf = multi_freq_obs(observations)
    mf.find_common_sources()
    mf.make_multi_freq_star()
    return mf
        
def load_observation(fp: str):
    """
    loads in an observation
    :param fp: the filepath and name of the file resulting from lightcurve_extraction.save_data
    :type fp: str
    """
    d = np.load(fp, allow_pickle=True)['arr_0'].flatten()[0]
    obs = observation(source=d['source'], full_frame_fns=d['full_frame_fns'], out_dir=d['out_dir'])
    for k in d.keys():
        obs.__dict__[k] = d[k]
    return obs

def load_source(fp: str):
   d = np.load(fp, allow_pickle=True)['arr_0'].flatten()[0]
   source = radio_source(label=d['label'], positions=d['positions'], avg_position=d['avg_position'], seps=d['seps'], pos_angs=d['pos_angs'], fluxes=d['fluxes'])
   for k in d.keys():
       source.__dict__[k] = d[k]
   return source

def load_frame(fp: str):
    d = np.load(fp, allow_pickle=True)['arr_0'].flatten()[0]
    frame = radio_source(file_name=d['file_name'], index=d['index'], working_directory=d['working_directory'])
    for k in d.keys():
        frame.__dict__[k] = d[k]
    return frame
  
            