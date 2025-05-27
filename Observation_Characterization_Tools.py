# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:04:39 2021

@author: iveli
"""

import numpy as np
from astropy import constants as const, units as u
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#%% Photometric Bands Class
fluxUnit = u.cm**-2 * u.s**-1 *u.Angstrom**-1
spectralIntensityUnits = 'erg*s**-1*cm**-2*steradian**-1*nm**-1';
intensityUnits = 'erg*s**-1*cm**-2*steradian**-1';
intensityUnitsMult = u.erg*u.s**-1*u.cm**-2*u.steradian**-1
luminosityUnitsMult = u.erg*u.s**-1
fluxUnitsMult = u.erg*u.s**-1*u.cm**-2
fluxUnits = 'erg*s**-1*cm**-2';
class PhotometricBands:
    """
    Class for instantiating the band information that will be used in photometric
    calculations. Acceptable inputs are u, g, r, i, U, B, V where u, g, r, and i are Sloan
    filters and U, B, and V are Johnson filters
    """
    def __init__(self,name = ''):
        self.name = name
        self.bandwidth = 0*u.nm
        self.centerLambda = 0*u.nm
        self.lensEfficiency = 1.
        self.mirrorEfficiency = 1.
        self.atmosphericTransfer = 1.
        self.nMirrors = 2
        self.totalEfficiency = self.lensEfficiency*self.mirrorEfficiency**self.nMirrors
        self.zeroPoint = 0.
        self.skyMag = 0
        self.__CalculateTotalEfficiency()
    def __AssignEfficiency(self):
        """
        Lens efficiency is from the silica lens numbers, not the quartz
        """
        name = self.name
        acceptableNames= ['u','g','r','i','U','B','V']
        if name == 'u':
            self.bandwidth = 65*u.nm
            self.centerLambda = 352.5*u.nm
            self.lensEfficiency = 0.86
            self.mirrorEfficiency = 0.75
            self.zeroPoint = 1539.3*fluxUnit
            self.skyMag = 22.454

        if name == 'g':
            self.bandwidth = 149*u.nm
            self.centerLambda = 475.5*u.nm
            self.lensEfficiency = 1.
            self.mirrorEfficiency = 0.94
            self.zeroPoint = 1134.6*fluxUnit
            self.skyMag = 22.22

        if name == 'r':
            self.bandwidth = (695-562)*u.nm
            self.centerLambda = 562*u.nm + (695-562)*u.nm/2
            self.lensEfficiency = 1. #Ivey check this
            self.mirrorEfficiency = 1. #Ivey check this
            self.zeroPoint = 875.4*fluxUnit
            self.skyMag = 21.34

        if name == 'i':
            self.bandwidth = (844 - 695)*u.nm
            self.centerLambda = 844*u.nm + (844 - 695)*u.nm/2
            self.lensEfficiency = 1. #Ivey check this
            self.mirrorEfficiency = 1. #Ivey check this
            self.zeroPoint = 714.5*fluxUnit
            self.skyMag = 20.45

        if name == 'U':
            self.bandwidth = 68*u.nm
            self.centerLambda = 365.6*u.nm
            self.lensEfficiency = 0.86
            self.mirrorEfficiency = 0.75
            self.zeroPoint = 756.1*fluxUnit
            self.skyMag = 22

        if name == 'B':
            self.bandwidth = 156.2*u.nm
            self.centerLambda = 435.3*u.nm
            self.lensEfficiency = 1.
            self.mirrorEfficiency = 0.92 
            self.zeroPoint = 632*fluxUnit
            self.skyMag = 22.7

        if name == 'V':
            self.bandwidth = 198.2*u.nm
            self.centerLambda = 547.7*u.nm
            self.lensEfficiency = 1.
            self.mirrorEfficiency = 0.94
            self.zeroPoint = 995.5*fluxUnit
            self.skyMag = 21.8

        elif name not in acceptableNames:
            warningStr = "Band name " +name +" not recognized, please use one of the following:\n"
            warningStr += 'u, g, r, i, U, B, V'
            raise Warning(warningStr)
    
    def __CalculateTotalEfficiency(self):
        self.__AssignEfficiency()
        self.totalEfficiency = self.mirrorEfficiency**self.nMirrors * self.atmosphericTransfer*self.lensEfficiency

#%%
def __TemperatureCheck(temperature):
    try:
        temperature.value
    except:
        #print('Assuming temperature is in Kelvin')
        temperature = temperature*u.K
    return temperature;

def __RadiusCheck(radius):
    try:
        radius.value
    except:
        print('Assuming stellar radius is in units of Solar radii')
        radius = radius*u.R_sun
    return radius

def __DistanceCheck(distance):
    try:
        distance.value
    except:
        print('Assuming distance to star is in units of pc')
        distance = distance*u.pc
    return distance



def __DiameterCheck(diameter):
    try:
        diameter.value
    except:
        print('Assuming aperture diameter is in units of meters')
        diameter = diameter*u.m
    return diameter


#%% Telescope Tools Class
class TelescopeTools:
    def __init__(self,band_names,diameter,elevation,pix_size,plate_scale):
        self.band_names = band_names
        self.bands = []
        for band_name in self.band_names:
            self.bands.append(PhotometricBands(band_name))
        self.diameter = __DiameterCheck(diameter)
        self.area = np.pi*(diameter/2)**2
        self.elevation = elevation
        self.pix_size = pix_size
        self.plate_scale = plate_scale
        return

#%% Flux Tools Class
class FluxTools:

    def __init__(self,bandName):
        self.bandName = bandName
        self.filter = PhotometricBands(bandName)

    def __TemperatureCheck(self,temperature):
        try:
            temperature.value
        except:
            #print('Assuming temperature is in Kelvin')
            temperature = temperature*u.K
        return temperature;

    def __RadiusCheck(self,radius):
        try:
            radius.value
        except:
            print('Assuming stellar radius is in units of Solar radii')
            radius = radius*u.R_sun
        return radius

    def __DistanceCheck(self,distance):
        try:
            distance.value
        except:
            print('Assuming distance to star is in units of pc')
            distance = distance*u.pc
        return distance

    def CalculateFluxDensity(self,mag):
        """
        Calculates photons per second, per Angstronm, per cm^2 given the magnitude of the object
        """
        zeroPoint = self.filter.zeroPoint
        return zeroPoint*10**(-mag/2.5)

    def CalculateFlux(self,mag,diameter):
        """
        Calculates photons per second. Diameter is assumed to be in meters unless
        specified otherwise
        """
        fluxDensity = self.CalculateFluxDensity(mag)
        bandwidth = self.filter.bandwidth
        efficiency = self.filter.totalEfficiency
        try:
            diameter = diameter.to('m')
        except:
            diameter = diameter*u.m
        area = np.pi*(diameter/2)**2
        return (fluxDensity*bandwidth*efficiency*area).to('1/s')

    def CalculateTotalPhotons(self,mag,diameter,intTime):
        """
        Calculates the total number of photons hitting the detector from a source
        in a given integration time. Assumes the telescope diameter is in meters
        and the integration time is in seconds unless specified otherwise
        """
        flux = self.CalculateFlux(mag,diameter)
        try:
            intTime = intTime.to('s')
        except:
            intTime = intTime*u.s
        return flux*intTime

    def CalculateRequiredIntTime(self,mag,diameter,nPhot):
        """
        Calculates the integration time required to acquire nPhot number of photons from a mag magnitude source
        :param mag: magnitude of the source
        :type mag: float
        :param diameter: diameter of the telescope primary
        :type diameter: astropy.units.quantity.Quantity
        :param nPhot: number of photons from the source you want to receive
        :type nPhot: float
        :return: integration time required to acquire nPhot number of photons from a mag magnitude source
        :rtype: astropy.units.quantity.Quantity

        """
        flux = self.CalculateFlux(mag,diameter)
        return nPhot/flux;

    def CalculateScintNoise(self,airMass,diameter,intTime):
        """
        Calculates the scintillation noise assuming Palomar elevation (h = 1712m)
        This is from equations 5 and 6 from the bible. Diameter should be given in
        CENTIMETERS!!
        """
        h = 1712*u.m
        h0 = 8000*u.m
        D = diameter
        try:
            D = D.to('cm')
        except:
            print('Assuming diameter of aperture is given in centimeters')
            D = D*u.cm
        chi = airMass
        t = intTime
        f = 0.09*D**(-2/3)*chi**(1.75)*(2*t)**(-1/2) * np.exp(-(h/h0).value)
        return (1.5 * 2**0.5 *f).value

    def CalculateSigPhoton(self,mag,diameter,intTime,nBGPix = 500,nPSF = 16,detectorDim = 1024,Gain = 1):
        skyMag = self.filter.skyMag
        readNoise = 12.
        readNoisePerPix = readNoise/detectorDim**2
        darkCurrentPerPixPerSec = 2.5*10**-4/u.s
        sourcePhotons = self.CalculateTotalPhotons(mag,diameter,intTime)
        skyPhotons = self.CalculateTotalPhotons(skyMag,diameter,intTime)
        nAperture = (nPSF/2)**2 * np.pi
        nPix = nAperture*(1 + nAperture/nBGPix)
        try:
            intTime = intTime.to('s')
        except:
            intTime = intTime*u.s
        Nsquared = sourcePhotons + skyPhotons*nPix
        Nsquared += nPix*intTime*darkCurrentPerPixPerSec
        Nsquared += nPix*readNoisePerPix #should readNoisePerPix be squared??
        
        sigPhot = Gain/Nsquared**(0.5)
        return sigPhot

    def CalculateScintToPhot(self,mag,diameter,intTime,airMass,nBGPix = 200, detectorDim = 1024, Gain = 1):
        sigPhot = self.CalculateSigPhoton(mag,diameter,intTime,nBGPix,detectorDim,Gain)
        sigScint = self.CalculateScintNoise(airMass,diameter,intTime)
        return sigScint/sigPhot

    def AirMassVsMag(self, mag,diameter,Gain = 1, A =1,nBGPix = 200,nPSF = 16):
        filt = self.filter
        nAperture = (nPSF/2)**2 * np.pi
        nPix = nAperture*(1 + nAperture/nBGPix)
        h = 1712*u.m
        h0 = 8000*u.m
        try:
            diameter = diameter.to('cm')
        except:
            diameter = diameter*u.cm
        fluxSource = self.CalculateFlux(mag,diameter)
        fluxSky = self.CalculateFlux(filt.skyMag,diameter)*nPix
        darkCurrent = 2.5*10**-4 * nPix*1/u.s
        N = np.sqrt(fluxSource + fluxSky + darkCurrent)**(-1)
        f = N/(1.5*0.09*diameter**(-2/3))*np.exp(h/h0)
        f *= Gain*A
        chi = f**(1/1.75)
        return chi.value

    def CalculateBolometricFlux(self,temperature):
        temperature = self.__TemperatureCheck(temperature)
        return (temperature**4)*const.sigma_sb
    
    def CalculateBolometricLuminosity(self,temperature,radius):
        temperature = self.__TemperatureCheck(temperature)
        radius = self.__RadiusCheck(radius)
        return ((temperature**4)*const.sigma_sb*4*np.pi*radius**2).to('erg/s')

    def BlackBodyModel(self,temperature,n =2**8):
        temperature = self.__TemperatureCheck(temperature)
        filt = self.filter
        center = filt.centerLambda
        minLambda = center -filt.bandwidth/2
        maxLambda = center + filt.bandwidth/2
        lambdaVec = np.linspace(minLambda,maxLambda,n)
        h = const.h
        c = const.c
        k = const.k_B
        numerator = 2 * h*c**2/lambdaVec**5
        denominator = np.exp((h*c/(lambdaVec*k*temperature)).to('')) -1
        spectralIntensity = (numerator/denominator)*u.steradian**-1
        return spectralIntensity.to(spectralIntensityUnits)

    def __BlackBodyIntegral(self, wavelength,temperature,includeTransmissivity = False):
        temperature = self.__TemperatureCheck(temperature)
        wavelength = wavelength*u.nm
        h = const.h
        c = const.c
        k = const.k_B
        numerator = 2 * h*c**2/wavelength**5
        denominator = np.exp((h*c/(wavelength*k*temperature)).to('')) -1
        spectralIntensity = (numerator/denominator)*u.steradian**-1
        if includeTransmissivity == True:
            if wavelength.value < 350:
                eff = (-np.exp(-(wavelength.value-350)/(wavelength.value-220))+ 2)*self.filter.totalEfficiency
                spectralIntensity = spectralIntensity*eff

            elif wavelength.value >= 350:
                eff = self.filter.totalEfficiency
                spectralIntensity = spectralIntensity*eff
            #print(eff)
        return spectralIntensity.to(spectralIntensityUnits).value

    def CalculateSolidAngle(self,radius,distance):
        radius = self.__RadiusCheck(radius)
        distance = self.__DistanceCheck(distance)
        return (radius**2/distance**2).to('')*np.pi*u.steradian

    def CalculateIntensityOverBand(self,temperature,includeTransmissivity = False):
        temperature = self.__TemperatureCheck(temperature)
        minLambda = (self.filter.centerLambda - self.filter.bandwidth/2).value
        maxLambda = (self.filter.centerLambda + self.filter.bandwidth/2).value
        intensity,toss = integrate.quad(self.__BlackBodyIntegral,minLambda,maxLambda,(temperature.value,includeTransmissivity))
        return intensity*intensityUnitsMult

    def CalculateLuminosityOverBand(self,temperature,radius,includeTransmissivity = False):
        radius = self.__RadiusCheck(radius)
        intensityOverBand = self.CalculateIntensityOverBand(temperature,includeTransmissivity)
        return (intensityOverBand*4*np.pi*u.steradian*radius**2).to('erg/s')

    def FluxtoLuminosity(self,flux,distance):
        distance = self.__DistanceCheck(distance)
        try:
            flux.value
        except:
            flux = flux*fluxUnitsMult
        return (flux*4*np.pi*distance**2).to('erg/s')

    def LuminosityToFlux(self,luminosity,distance):
        distance = self.__DistanceCheck(distance)
        try:
            luminosity.value
        except:
            luminosity = luminosity*luminosityUnitsMult
        flux = (luminosity/(4 * np.pi*distance**2)).to('erg/(s*m**2)')
        return flux

    def CalculateContrast(self,quietTemp,flareTemp,radius,flare_luminosity,includeTransmissivity = False):
        try:
            flare_luminosity.value
        except:
            flare_luminosity = flare_luminosity*luminosityUnitsMult
        flareLumBol = self.CalculateBolometricLuminosity(flareTemp,radius)
        """
        NOTE: This will be kinda inaccurate because you can't really include
        the transmissivity of the atmosphere in the bolometeric luminosity
        calculation
        """
        flareLumBand = self.CalculateLuminosityOverBand(flareTemp,radius,includeTransmissivity) # Calculate flare luminosity in the band as if it were the size of the star
        quietLumBand = self.CalculateLuminosityOverBand(quietTemp,radius,includeTransmissivity) # Calculate full-disk, quiet stellar luminosity in the band
        
        fractionalFlareAreaBol = flare_luminosity/flareLumBol # determine area of flare based on its assigned bolometric luminosity and what it would be if it were the size of the star
        fractionalFlareLum = flareLumBand*fractionalFlareAreaBol # Multiply the flare, full-disk luminosity in the band by the fractional area it makes up on the star
        return fractionalFlareLum/quietLumBand
    
    
    def CalculateDiffusionSize(self,mag,diameter,intTime,well_depth = 40_000):
        totalPhotons = self.CalculateTotalPhotons(mag, diameter, intTime)
        n_pixels = totalPhotons/well_depth
        return n_pixels
    
    def CalculateDiffuserDistance(self,mag,diameter,intTime,well_depth,diff_full_angle = 1*u.deg,pix_size = 13*u.micron):
        n_pix = self.CalculateDiffusionSize(mag, diameter, intTime)
        r = int(np.ceil((n_pix/np.pi)**0.5))
        r_phys = r*pix_size
        half_angle = diff_full_angle/2
        dist = (r_phys/np.tan(half_angle)).to('cm')
        return dist
    
    
    def CalculateLimitingMagnitude(self, well_depth, int_time, diameter):
        bandwidth = self.filter.bandwidth
        phi0 = self.filter.zeroPoint
        area = np.pi*(diameter/2)**2
        max_val = well_depth/2
        mag = -2.5 * np.log10(max_val/(phi0*area*int_time*bandwidth))
        return mag
    
    def CalculateMagFromFlux(self, flux, int_time, diameter =0.5*u.m, eff = 0.11):
        if eff == None:
            eff = self.filter.totalEfficiency
        area =  np.pi*(diameter/2)**2
        arg = (flux/eff/area/int_time/self.filter.bandwidth/self.filter.zeroPoint).to('')
        m = -2.5 * np.log10(arg)
        return m
    

        
        
def CalculategPrime(B,V):
    return V + 0.56*(B-V) - 0.12

#%% Redo of noise calculation
def NoisePerFrame(D, nPix, readoutNoise = 6, magVec = np.linspace(3,15,100),frameRate = 1/u.s,chi = 1.5,pixelScale =None, expTime = None):
    try:
        D = D.to('m')
    except:
        D = D*u.m
        print('No units given for telescope diameter; assuming meters')
    
    gBandTools = FluxTools('g')
    if pixelScale == None:
        pixelScale = (206265*u.arcsecond/(6.8*D) * 13*u.micron).to('arcsecond').value
    expTime = 1/frameRate

    starNoise = gBandTools.CalculateTotalPhotons(magVec,D,expTime)
    skyNoise = gBandTools.CalculateTotalPhotons(gBandTools.filter.skyMag*pixelScale**2,D,expTime)
    readNoise = readoutNoise * np.sqrt(nPix)
    # print(readNoise)
    ppm = 10**6
    photNoise = np.sqrt(starNoise**2 + skyNoise**2 +readNoise**2)
    sigScint = gBandTools.CalculateScintNoise(chi,D,expTime)
    # totalNoise = ((sigScint*ppm)**2 + (photNoise*ppm/starPhotons)**2)**0.5
    return starNoise,skyNoise,readNoise,sigScint
    


#%% Noise Calculation 
def CalculateNoise(D, integrationTime, nPix, readoutNoise = 6, magVec = np.linspace(3,14,100),frameRate = 1/u.s,chi = 1.5,pixelScale =None):
    try:
        D = D.to('m')
    except:
        D = D*u.m
        print('No units given for telescope diameter; assuming meters')
    try:
        integrationTime = integrationTime.to('s')
    except:
        integrationTime = integrationTime*u.s
    try:
        frameRate.value
    except:
        frameRate = frameRate/u.s
    gBandTools = FluxTools('g')
    if pixelScale == None:
        pixelScale = (206265*u.arcsecond/(6.8*D) * 13*u.micron).to('arcsecond').value
    nFrames = integrationTime*frameRate
    
    starPhotons = gBandTools.CalculateTotalPhotons(magVec,D,integrationTime/nFrames)
    skyPhotons = gBandTools.CalculateTotalPhotons(gBandTools.filter.skyMag*pixelScale**2,D,integrationTime/nFrames)
    
    starNoise = np.sqrt(starPhotons/nFrames)
    skyNoise = np.sqrt(skyPhotons*nPix/nFrames) 
    readNoise = readoutNoise * np.sqrt(nPix/nFrames)
    # print(readNoise)
    ppm = 10**6
    photNoise = np.sqrt(starNoise**2 + skyNoise**2 +readNoise**2)
    sigScint = gBandTools.CalculateScintNoise(chi,D,integrationTime/nFrames)/np.sqrt(nFrames)
    totalNoise = ((sigScint*ppm)**2 + (photNoise*ppm/starPhotons)**2)**0.5
    
    A = np.pi*(D/2)**2
    eff = gBandTools.filter.totalEfficiency
    dlam = gBandTools.filter.bandwidth
    dt = 1*u.s
    phi0 = gBandTools.filter.zeroPoint
    halfFull = 45000
    maxMag = np.log10((halfFull*nPix/phi0/A/eff/dt/dlam).to(''))*-2.5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.semilogy(magVec, starNoise/starPhotons*ppm,color = 'darkorange',label = 'Star Noise')
    plt.semilogy(magVec,skyNoise/starPhotons *ppm,color = 'mediumvioletred', label = 'Sky Noise')
    plt.semilogy(magVec, readNoise/starPhotons *ppm,color = 'deepskyblue',label = 'Read Noise')
    plt.semilogy(magVec, photNoise/starPhotons *ppm,color = 'yellowgreen',label = 'Photometric Noise')
    plt.semilogy(magVec, np.linspace(sigScint,sigScint,len(magVec))*ppm,label = 'Scintillation Noise')
    plt.semilogy(magVec, totalNoise ,color = 'black',linewidth=2,label = 'Total Noise')

    # plt.plot(np.linspace(maxMag,maxMag),np.linspace(min(skyNoise/starPhotons*ppm),max(totalNoise)),'--k')
    mmag = (10**(10**-3/2.5) -1)*10**6
    anchorx,anchory = magVec[0],readNoise/starPhotons[0] *ppm
    rect = Rectangle((anchorx,anchory),magVec[-1]-magVec[0],mmag,alpha = 0.3)
    rect_saturation = Rectangle((maxMag,anchory), magVec[-1]-maxMag,totalNoise[-1], alpha = 0.3,color = 'y')
    #ax.add_patch(rect)
    #ax.add_patch(rect_saturation)
    
    plt.xlabel('Star Magnitude',fontsize = 14)
    plt.ylabel(r'Noise Contribution [ppm]',fontsize = 14)
    plt.title(r'$n_{pix}$ = '+str(int(nPix))+r', $\chi$ = '+str(chi)+', # Frames = '+str(int(nFrames))+', D = '+str(D),fontsize=14)
    plt.legend(fontsize = 12)
    
    plt.xlim((magVec.min(),magVec.max()))
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid('on')
    plt.show()
    plt.tight_layout()
    return totalNoise
#%% Plotting Tools
plt.rcParams.update({'font.family':'calibri'})
def PlotScintVPhot(diameter = 0.5*u.m, band:str = 'g',magVec = np.linspace(5,15,10**3),chiVec = np.linspace(1,2,10**3),intTime=1):
    plt.rcParams.update({'font.family':'calibri'})
    chi,mag = np.meshgrid(chiVec,magVec)
    chi,mag = np.meshgrid(chiVec,magVec)
    
    photTools = FluxTools(band)
    sigPhot = photTools.CalculateSigPhoton(mag, diameter, intTime)
    sigScint = photTools.CalculateScintNoise(chi, diameter, intTime)
    ratio = sigScint/sigPhot
    
    chiFunc =photTools.AirMassVsMag(magVec,diameter)
    chiFunc10 = photTools.AirMassVsMag(magVec,diameter,A=10)
    plt.figure()
    plt.imshow(np.log10(ratio),extent=[min(chiVec),max(chiVec),min(magVec),max(magVec)],aspect = 'auto',origin = 'lower')
    cbar =plt.colorbar()
    cbar.set_label(label = r"$\log{(\sigma_{S}/\sigma_{phot})}$",size = 14)
    plt.plot(chiFunc,magVec,'w',label = r'$\sigma_{scint}/\sigma_{phot} = 1$')
    plt.plot(chiFunc10,magVec,'--w',label = r'$\sigma_{scint}/\sigma_{phot} = 10$')
    plt.ylim([magVec[0],magVec[-1]])
    plt.xlim([chiVec[0],chiVec[-1]])
    
    plt.xlabel(r'$\chi$',fontsize = 14)
    plt.ylabel('Magnitude',fontsize = 14)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.legend()
    plt.tight_layout()
    return

def CalculateDeltaMagFromContrast(x):
    dmag = -2.5*np.log10(1+x)
    print(x)
    print(dmag)
    return -2.5*np.log10(1+x)

def CalculateContrastFromDeltaMag(x):
    return 10**(-x/2.5) -1


def CalculateFillFactor(x,Tq = 5080, Tf = 9000):
    fill = x * (Tq/Tf)**4
    return fill

def CalculateFluxRatioFromFill(x,Tq = 5080,Tf = 9000):
    Lf = x * (Tf/Tq)**4 
    return Lf
    

def PlotContrasts(quietTemp = 5080*u.K, flareTemp = 10000*u.K,R = 0.735*const.R_sun, bolometricLum =np.linspace(10**30,10**34,1000)*u.erg/u.s ):
    plt.rcParams.update({'font.family':'calibri'})
    c_names = ['blue','green','red','maroon','mediumslateblue','teal','springgreen']
    uBandTools = FluxTools('u')
    gBandTools = FluxTools('g')
    rBandTools = FluxTools('r')
    iBandTools = FluxTools('i')
    #UBandTools = FluxTools('U')
    #BBandTools = FluxTools('B')
    #VBandTools = FluxTools('V')

    uContrast = uBandTools.CalculateContrast(quietTemp,flareTemp,R,bolometricLum,True)
    gContrast = gBandTools.CalculateContrast(quietTemp,flareTemp,R,bolometricLum,True)
    rContrast = rBandTools.CalculateContrast(quietTemp,flareTemp,R,bolometricLum,True)
    iContrast = iBandTools.CalculateContrast(quietTemp,flareTemp,R,bolometricLum,True)
    #UContrast = UBandTools.CalculateContrast(quietTemp,flareTemp,R,bolometricLum,True)
    #BContrast = BBandTools.CalculateContrast(quietTemp,flareTemp,R,bolometricLum,True)
    #VContrast = VBandTools.CalculateContrast(quietTemp,flareTemp,R,bolometricLum,True)

    fig,ax = plt.subplots()
    # ax2 = ax.twinx()
    #ax3 = ax.twiny()
    
    #ax2 = ax.secondary_yaxis('right',functions = (CalculateDeltaMagFromContrast,CalculateContrastFromDeltaMag))
    
    x = np.linspace(min(iContrast),max(uContrast),len(bolometricLum))
    delta_mags = 2.5*np.log10(1+x)
    # print(delta_mags[0])
    # print(delta_mags[-1])
    # ax2.loglog(bolometricLum,delta_mags,alpha = 0)
    # #ax2.set_ylim(delta_mags[0],delta_mags[-1])
    # ax2.set_ylabel(r'$\log(\Delta$mag)',fontsize = 15)
    # ax2.tick_params(axis = 'both',labelsize = 13)
    
    
    ax.loglog(bolometricLum,uContrast,color = c_names[0],label = "u' band")
    ax.loglog(bolometricLum,gContrast,color = c_names[1],label = "g' band")
    ax.loglog(bolometricLum,rContrast,color = c_names[2],label = "r' band")
    ax.loglog(bolometricLum,iContrast,color = c_names[3],label = "i' band")
    #ax.loglog(bolometricLum,UContrast,color = c_names[4],label = "U band")
    #ax.loglog(bolometricLum,BContrast,color = c_names[5],label = "B band")
    #ax.loglog(bolometricLum,VContrast,color = c_names[6],label = "V band")
    
    
    L= (4*np.pi*R**2 * const.sigma_sb*(quietTemp**4)).to('erg/s')
    ax.loglog(bolometricLum,bolometricLum/L,color = 'k',label= 'Bolometric contrast')
    ax.plot(np.linspace(L,L,100),np.linspace(min(iContrast),max(uContrast),100),'k--',label=r'Quiescent L$_{bol}$')
    
    #fillFactors = bolometricLum/L * (quietTemp/flareTemp)**4
    #ax3.loglog(fillFactors,bolometricLum,alpha = 0)
    #ax3.set_xlabel('Fill factor',fontsize = 15)
    #ax3.tick_params(axis = 'both',labelsize = 13)

    #ax.set_title(r'Contrasts T$_{quiet}$ = '+str(int(quietTemp.value))+'K, R = '+str(float(R.to('R_sun').value))+r'R$_\odot$',fontsize = 15)
    #ax.set_title(r'Contrasts for T$_{quiet}$ = '+str(int(quietTemp.value))+'K, T$_{flare}$ = '+str(int(flareTemp.value))+'K',fontsize = 15)
    ax.set_xlabel(r'L$_{bol}$ [erg/s]',fontsize = 15)
    ax.set_ylabel(r'L$_{band,flare}$/L$_{band,quiet}$',fontsize = 15)
    ax.tick_params(axis = 'both',labelsize = 13)
    ax.set_ylim(min(iContrast),max(uContrast))

    ax.legend()
    ax.grid('on')
    ax.set_xlim((bolometricLum[0].value,bolometricLum[-1].value))
    plt.tight_layout()
    return

def PlotScintNoise(acceptableNoise,diameter= 0.5*u.m,tvec= np.linspace(1,60*180,100), chivec = np.linspace(1,2,100)):
    plt.rcParams.update({'font.family':'calibri'})
    t,chi = np.meshgrid(tvec,chivec)
    photTools = FluxTools('g')
    sigScint = photTools.CalculateScintNoise(chi,diameter,t)
    
    palomarEl = 1712*u.m
    h0 = 8000*u.m
    plt.figure()
    plt.title(r'Scintillation Noise',fontsize = 14)
    plt.imshow(np.log10(sigScint),extent=[min(tvec),max(tvec),max(chivec),min(chivec)],aspect='auto')
    cb = plt.colorbar()
    cb.set_label(label= r'$\log{(\sigma_S)}$',size =14)
    if type(acceptableNoise) is list:
        for i in range(len(acceptableNoise)):
            n = acceptableNoise[i]
            print(n)
            airmass = n*tvec**0.5 *diameter.to('cm')**(2/3)/(1.5*0.09)*np.exp(palomarEl/h0)
            airmass = airmass**(1./1.75)
            plt.plot(tvec,airmass,'w',linewidth = 0.75*(i+1),label = r'$\sigma = $'+str(n))

    else:
        airmass = acceptableNoise*tvec**0.5 *diameter.to('cm')**(2/3)/(1.5*0.09)*np.exp(palomarEl/h0)
        airmass = airmass**(1./1.75)
        plt.plot(tvec,airmass,'w',label = r'$\sigma = $'+str(acceptableNoise))

    plt.xlabel(r'$t_{int}$ [s]',fontsize = 14)
    plt.ylabel(r'$\chi$',fontsize = 14)
    plt.ylim(min(chivec),max(chivec))
    plt.xticks(fontsize = 13,rotation = 40)
    plt.yticks(fontsize = 13)
    plt.legend()
    plt.tight_layout()
    return airmass


