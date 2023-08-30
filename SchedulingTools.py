# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:02:30 2022

@author: Ivey
"""

import numpy as np;
from astropy import units as u
from astropy.coordinates import get_sun, get_moon
from astropy.coordinates import AltAz, SkyCoord
from astroquery.simbad import Simbad as Simbad
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import EarthLocation
import matplotlib.pyplot as plt
import warnings

long = -116.84*u.deg
lat = 33.36*u.deg
utc_offset = 7*u.hr
pal_loc = EarthLocation(lat = lat, lon = long)
ovro_loc = EarthLocation(lat = 37.2317*u.deg, lon = -118.2951*u.deg)
"""
all of my if not statements for the utc time can be a function call instead
"""
#%%

def AccessSIMBAD(objID:str):
    """
    Returns SkyCoords object of the target including RA, Dec, and proper motions
    :param objID: The name of the target object
    :type objID: str
    :return: The SkyCoord object with RA, Dec, and proper motions
    :rtype: SkyCoord

    """
    if objID == None:
        print('You need to give an object name')
    try:
        Simbad.add_votable_fields('main_id','propermotions')
        simbadResult = Simbad.query_object(objID)[0]
    except:
        print('Unable to access the object from SIMBAD.')
        return;
    try:
        dist_val = simbadResult['Distance_distance']*u.pc
        dist_unit = str(simbadResult['Distance_unit'])
        dist = dist_val.to(dist_unit)
    except:
        warnings.warn('Unable to determine distance of object '+objID)
        dist = 100*u.pc
    obj_ra_str = simbadResult['RA'] 
    obj_dec_str = simbadResult['DEC']
    obj_ra_mu = simbadResult['PMRA'] * u.mas/u.yr
    obj_dec_mu = simbadResult['PMDEC'] * u.mas/u.yr
    obs_time = "J2000"
    coord_str = obj_ra_str + ' ' + obj_dec_str
    coords = SkyCoord(coord_str,distance = dist,frame = 'icrs', unit = (u.hourangle, u.deg), obstime = obs_time, pm_ra_cosdec = obj_ra_mu, pm_dec = obj_dec_mu)
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
    return

def DecDegToTriplet(coords: SkyCoord):
    """
    Converts declination degree from to [degree, arcmin, arcsec] form
    :param coords: Coordinates of object to convert
    :type coords: SkyCoord
    :return: 3-value representation of declination
    :rtype: list of floats
    
    """
    dec = coords.dec
    if dec < 0:
        deg = abs(int(dec.deg))
        arcmin = abs(int(dec.arcmin + deg * 60))
        arcsec = abs(dec.arcsec + deg*3600 + arcmin*60)
        return [-deg,arcmin,arcsec]
    if dec > 0:
        deg = int(dec.deg)
        arcmin = int(dec.arcmin - deg * 60)
        arcsec = (dec.arcsec - deg*3600 - arcmin*60)
        return [deg,arcmin,arcsec]
        

def CalculateAltAz(obj: SkyCoord, obstime: Time,obstime_is_utc = False, loc = pal_loc):
    """
    Calculates altitude and azimuth of an object at a given observing time assuming observing from Palomar
    :param obj: Coordinates of the target (object) of interest
    :type obj: SkyCoord
    :param obstime: Datetime of when you want to observe the target
    :type obstime: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: Bool, optional
    :param loc: Location to be observing the target from. Default is Palomar
    :type loc: EarthLocation, optional
    :raises Exception: Occurs if it can't transform the object to the Palomar observing frame
    :return: Target object with alt-az information
    :rtype: SkyCoord


    """
    if obstime_is_utc == False:
        obstime += utc_offset
    aa = AltAz(obstime = obstime, location = loc)
    try:
        obj_altaz = obj.transform_to(aa)
        return obj_altaz
    except:
        raise Exception('Could not transform to alt-az coordinates')
    return

def SetAltAzFrame(obstime: Time, obstime_is_utc= False):
    """
    Sets the alt-az, Palomar observing frame based on observing time
    :param obstime: Observing date and time
    :type obstime: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: Bool, optional
    :return: Alt-az observing frame
    :rtype: AltAz

    """
    if not obstime_is_utc:
        obstime+= utc_offset
    aa = AltAz(obstime = obstime, location = pal_loc)
    return aa

def PlotAltitude(obj:SkyCoord, obstime:Time,obstime_is_utc = False,plot_sun =True, plot_moon = True,object_name = 'obj', obs_frame_aa = None):
    """
    Plots the altitude wrt time of an object for an observing night.
    :param obj: Coordiantes of the target
    :type obj: SkyCoord
    :param obstime: Observing date to plot the altitudes for
    :type obstime: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: Bool, optional
    :param plot_sun: Plots the location of the sun in the sky if True, defaults to True
    :type plot_sun: Bool, optional
    :param plot_moon: Plots the location of the moon in the sky if True, defaults to True
    :type plot_moon: Bool, optional
    :param object_name: Names of the target objects, defaults to 'obj'
    :type object_name: str, optional
    :param obs_frame_aa: Alt-az observing frame for the objects, defaults to None
    :type obs_frame_aa: AltAz, optional
    :return: None
    :rtype: None

    """
    if obstime_is_utc == False:
        obstime += utc_offset
    obsdate_midnight_str = obstime.datetime.date().isoformat() + ' 00:00:00'
    obsdate_midnight = Time(obsdate_midnight_str,format = 'iso')+utc_offset
    delta_midnight = np.linspace(-12,12,1000)*u.hr
    obs_times = obsdate_midnight+delta_midnight
    if obs_frame_aa is None:
        obs_frame_aa = AltAz(obstime = obs_times,location = pal_loc)
    obj_aa = obj.transform_to(obs_frame_aa)
    plt.plot(delta_midnight,obj_aa.alt,label = object_name)
    plt.title(obsdate_midnight.datetime.date().isoformat())
    plt.xlabel('Time from midnight [hr]')
    plt.ylabel('Altitude [deg]')
    if plot_sun == True:
        sun= get_sun(obs_times)
        sun_aa = sun.transform_to(obs_frame_aa)
        plt.plot(delta_midnight,sun_aa.alt,c = 'r',label = 'Sun')
    if plot_moon == True:
        moon = get_moon(obs_times)
        moon_aa = moon.transform_to(obs_frame_aa)
        plt.plot(delta_midnight,moon_aa.alt, c= 'k',label = 'Moon')
    plt.legend()
    return



def CheckSunAltitude(obs_time:Time,obstime_is_utc:bool= False,obs_frame_aa:AltAz = None):
    """
    Finds the Sun's elevation based on the observing time, assuming observing at Palomar
    :param obs_time: Time of the observation
    :type obs_time: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: Bool, optional
    :param obs_frame_aa: Alt-az observing frame for the objects, defaults to None
    :type obs_frame_aa: AltAz, optional
    :return: Altitude of the sun
    :rtype: float

    """
    if not obstime_is_utc:
        obs_time+=utc_offset
    sun= get_sun(obs_time)
    if obs_frame_aa is None:
        obs_frame_aa = AltAz(obstime = obs_time,location = pal_loc)
    sun_aa = sun.transform_to(obs_frame_aa)
    return sun_aa.alt

def CheckSunSet(obs_time:Time,obstime_is_utc= False, set_elevation = -5*u.deg):
    """
    Checks if the sun has set at a given time assuming observing at Palomar
    :param obs_time: Time of the observation
    :type obs_time: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: Bool, optional
    :param set_elevation: Minimum elevation for the sun to be considered set, defaults to -5*u.deg
    :type set_elevation: float, astropy unit, optional
    :return: Description whether the sun is set (True) or not (False)
    :rtype: Bool

    """
    sun_alt = CheckSunAltitude(obs_time,obstime_is_utc)
    if sun_alt > set_elevation:
        return False
    if sun_alt <= set_elevation:
        return True
    return

def GetMoonLocation(obs_time:Time,obstime_is_utc= False,obs_frame_aa = None):
    """
    Finds altitude and azimuth of the moon at a given datetime assuming observing from Palomar
    :param obs_time: Date and time of the observation
    :type obs_time: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: Bool, optional
    :param obs_frame_aa: AltAz observing frame, defaults to None
    :type obs_frame_aa: AltAz, optional
    :return: Coordinates of moon, including altitude and azimuth at the time of the observation
    :rtype: SkyCoord

    """
    if not obstime_is_utc:
        obs_time+=utc_offset
    moon= get_moon(obs_time)
    if obs_frame_aa is None:
        obs_frame_aa = AltAz(obstime = obs_time,location = pal_loc)
    moon_aa = moon.transform_to(obs_frame_aa)
    return moon_aa

def FindDistanceFromMoon(obj:SkyCoord, obs_time,obstime_is_utc = False, obs_frame_aa = None):
    """
    Finds the distance between a target and the moon at a given date and time
    :param obj: Target object
    :type obj: SkyCoord
    :param obs_time: Observing date and time of the object
    :type obs_time: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: Bool, optional
    :param obs_frame_aa: AltAz observing frame, defaults to None
    :type obs_frame_aa: AltAz, optional
    :return: Distance between the moon and target of interest in degrees
    :rtype: Float

    """
    if not obstime_is_utc:
        obs_time+= utc_offset
    if obs_frame_aa is None:
        obs_frame_aa = AltAz(obstime = obs_time,location = pal_loc)
    moon_aa = GetMoonLocation(obs_time, obstime_is_utc,obs_frame_aa)
    obj_aa = obj.transform_to(obs_frame_aa)
    d_alt = moon_aa.alt - obj_aa.alt
    d_az = moon_aa.az - obj_aa.az
    d = np.sqrt(d_alt**2 + d_az**2)
    return d

def CheckFarEnoughFromMoon(obj:SkyCoord, obs_time, obstime_is_utc = False, obs_frame_aa = None,min_distance = 10*u.deg):
    """
    Checks if an object is far enough from the moon to be safe to observe
    :param obj: Target Object
    :type obj: SkyCoord
    :param obs_time: Date and time of the observation of the target object
    :type obs_time: Time
    :param obstime_is_utc: Describes whether the observation time is UTC or not (False if not), defaults to False
    :type obstime_is_utc: bool, optional
    :param obs_frame_aa: AltAz observing frame, defaults to None
    :type obs_frame_aa: AltAz, optional
    :param min_distance: Minimum allowable distance between the target and the moon, defaults to 10*u.deg
    :type min_distance: Float, optional
    :return: Truth statement as to whether the object is far enough from the moon to safely observe (True)
    :rtype: bool

    """
    if obs_frame_aa is None:
        obs_frame_aa = AltAz(obstime = obs_time,location = pal_loc)
    d_moon = FindDistanceFromMoon(obj, obs_time, obstime_is_utc,obs_frame_aa)
    if d_moon < min_distance:
        return False
    if d_moon > min_distance:
        return True
    return

def FindTelescopePointingSettings(target:SkyCoord,companion:SkyCoord):
    """
    Finds the pointing location (midpoint between the target and companion) and suggested position angle of the CCD
    to fit both objects.
    :param target: Coordinates of the science target object
    :type target: SkyCoord
    :param companion: Coordiantes of the photometric companion of the science target
    :type companion: SkyCoord
    :return: Midpoint location and suggested position angle of the CCD
    :rtype: (SkyCoord, Angle)

    """
    sep = target.separation(companion)
    pa = target.position_angle(companion).to('deg')
    #NOTE: PA is measured east of north
    mp = target.directional_offset_by(pa, sep/2)
    # if sep > 13.7*np.sqrt(2)*u.arcmin:
    #     raise Exception('WARNING: Distance between target and companion is larger than can be accomodated by CCD')
    return mp, pa

class ObservationTarget:
    def __init__(self,name: str,companion:str,date: Time, priority:int,observingtime_is_utc = False,obs_frame_aa =None,filt = None):
        self.name = name
        self.companion_name = companion
        self.observingtime_is_utc= observingtime_is_utc
        self.priority = priority
        if obs_frame_aa is None:
            self.obs_frame_aa = AltAz(obstime = date,location = pal_loc)
        else:
            self.obs_frame_aa = obs_frame_aa
        if not filt:
            self.filt = 'g0'
        else:
            self.filt = filt
        target_coords = AccessSIMBAD(name)
        companion_coords = AccessSIMBAD(companion)
        self.target_coords = ApplyProperMotion(target_coords,date)
        self.companion_coords = ApplyProperMotion(companion_coords, date)
        mp, pa = FindTelescopePointingSettings(self.target_coords, self.companion_coords) 
        self.mp = mp[0]
        self.pa = pa[0]
        self.observing_date = date
        self.AssignValues()
        self.FlagObservingTimes()
    
    def AssignValues(self):
        aa = self.target_coords.transform_to(self.obs_frame_aa)
        self.altitudes = aa.alt
        self.azimuths = aa.az
        
    def FlagObservingTimes(self,minimum_el = 20*u.deg,minimum_distance_from_moon = 10*u.deg):
        distances = FindDistanceFromMoon(self.target_coords,obs_time=self.observing_date,obstime_is_utc=True,obs_frame_aa=self.obs_frame_aa)
        try:
            flag_array = np.empty(len(self.altitudes),dtype = bool)
        except:
            flag_array = np.empty(1,dtype = bool)
        for i in range(len(flag_array)):
            if len(flag_array)==1:
                d= distances
                altitude = self.altitudes
            else:
                d = distances[i]
                altitude = self.altitudes[i]
            if d < minimum_distance_from_moon or altitude < minimum_el:
                flag_array[i] = True
            if d > minimum_distance_from_moon and altitude > minimum_el:
                flag_array[i] = False
        self.flagged_observing_times = flag_array
            
        
class ObservationSchedule:
    def __init__(self,obj_names,companion_names, obs_date:Time,priorities,filts = None,observingtime_is_utc = False):
        if not observingtime_is_utc:
            obs_date += utc_offset
        self.obs_date = obs_date
        if not filts:
            self.filts = ['g0']*len(obj_names)
        else:
            self.filts = filts
        self.obj_names = obj_names
        self.comp_names = companion_names
        self.__UpdateValues()
        self.MakeObservingObjects(priorities)
        self.observing_schedule = None
        return
    def __UpdateValues(self):
        time_chunks = np.arange(-3.5,4.5,0.25)*u.hr
        obsdate_midnight_str = self.obs_date.datetime.date().isoformat() + ' 00:00:00'
        obsdate_midnight = Time(obsdate_midnight_str,format = 'iso') +utc_offset
        obs_chunks = obsdate_midnight + time_chunks
        
        self.observation_times = obs_chunks
        self.observing_table = Table(names = ['Observing_Time','Object_Name','MP_Coords','PA','Filter','Altitude'],dtype =[object,object,object,float,object,float])
        self.aas = SetAltAzFrame(obs_chunks,obstime_is_utc=True)
        return
    def MakeObservingObjects(self,priorities):
        self.objects = []
        for i in range(len(self.obj_names)):
            name = self.obj_names[i]
            comp = self.comp_names[i]
            filt = self.filts[i]
            self.objects.append(ObservationTarget(name,comp,self.observation_times,priority = priorities[i],observingtime_is_utc=True,filt = filt))
        return
    def PlotAllObjectAltitudes(self):
        plt.figure()
        time_chunks = np.arange(-3.5,4.5,0.25)*u.hr
        for obj in self.objects:
            altitudes = obj.altitudes
            plt.plot(time_chunks,altitudes,label = obj.name)
        moon_aa = GetMoonLocation(self.observation_times,True)
        plt.plot(time_chunks,moon_aa.alt,'--k',label = 'Moon')
        plt.title('UTC '+ self.obs_date.datetime.date().isoformat())
        plt.xlabel('Time from local midnight [hr]')
        plt.ylabel('Altitude [deg]')
        plt.grid('on')
        plt.ylim([0,90])
        plt.plot(np.linspace(-4,4),np.linspace(30,30),c='k')
        plt.legend()
        return
    def MakeObservingSchedule(self):
        for i in range(len(self.observation_times)):
            t = self.observation_times[i]
            unflagged_objects = []
            for o in self.objects:
                if not o.flagged_observing_times[i]: #i.e. if the flag is set to False
                    unflagged_objects.append(o)
            if len(unflagged_objects) == 0:
                r = [t,'',SkyCoord(ra = 0*u.deg,dec =0*u.deg),None]
            if len(unflagged_objects) == 1:
                r = [t,unflagged_objects[0].name,unflagged_objects[0].mp,unflagged_objects[0].pa,unflagged_objects[0].filt,unflagged_objects[0].altitudes[i].value]
            elif len(unflagged_objects) > 1:
                alt_tab = Table(names = ['Observing_Time','Object_Name','MP_Coords','PA','Filter','Altitude','Priorities'],dtype =[object,object,object,float,object,float,int])
                for unf in unflagged_objects:
                    row = [t,unf.name,unf.mp,unf.pa,unf.filt,unf.altitudes[i],unf.priority]
                    alt_tab.add_row(row)
                low_priority = np.where(alt_tab['Priorities'] != np.min(alt_tab['Priorities']))[0]
                alt_tab.remove_rows(low_priority)
                alt_tab.sort('Altitude')
                alt_tab.reverse()
                alt_tab.remove_column('Priorities')
                r = alt_tab[0]  
            try:
                self.observing_table.add_row(r)
            except:
                print('unable to add row:\n'+str(r))
        return
    
    def ConsolidateObservingSchedule(self):
        if len(self.observing_table) == 0:
            self.MakeObservingSchedule()
        observing_table = self.observing_table
        obj_names,i = np.unique(observing_table['Object_Name'],return_index = True)
        names_sort = [x for _,x in sorted(zip(i,obj_names))]
        tab = Table(names = ['Object_Name','Coordinates','PA','Filter','Start_Datetime','End_Datetime'],dtype = [object,object,float,object,object,object])
        try:
            names_sort.remove('')
        except:
            pass
        for name in names_sort:
            inds = np.where(observing_table['Object_Name'] == name)[0]
            inds.sort()
            start_time =  observing_table['Observing_Time'][inds[0]]
            end_time =  observing_table['Observing_Time'][inds[-1]]
            pa = observing_table['PA'][inds[0]]
            filt = observing_table['Filter'][inds[0]]
            coordinates = observing_table[inds[0]]['MP_Coords']
            row = [name,coordinates,pa,filt,start_time,end_time]
            tab.add_row(row)
        self.observing_schedule = tab
        
#%%
if __name__ == "__main__":
    t = Time('2022-12-13T17:01:00.0',format = 'isot') #+utc_offset
    # targets = ['epsilon eridani','kappa1 ceti','ek dra','pi1 uma','chi1 ori', 'EQ Peg','Cas A']
    targets = ['NGC 7662','M51','Rosette Nebula','Crab Nebula','Epsilon Eridani','Chi1 Ori','Trapezium Cluster']
    companions = targets
    # companions = ['HD 22130','BD+02 521','HD 129390','TYC 4133-776-1','HD 39417','BD+18 5161','Cas A']
    filts = ['g1','g1','g025','g1','g1','g0','g0']
    priorities = [0,1,1,2,2,3,3]
    obs = ObservationSchedule(targets, companions, obs_date = t, priorities = priorities)
    obs.MakeObservingSchedule()
    obs.ConsolidateObservingSchedule()
    obs.PlotAllObjectAltitudes()
#%%

class ObservingPlan:
    def __init__(self, observation_schedule: ObservationSchedule,flats_plan_fn,start_time:Time):
        if len(observation_schedule.observing_table) == 0:
            observation_schedule.MakeObservingSchedule()
            observation_schedule.ConsolidateObservingSchedule()
        if len(observation_schedule.observing_table) != 0 and not observation_schedule.observing_schedule:
            observation_schedule.ConsolidateObservingSchedule()
        self.schedule_table = observation_schedule.observing_schedule
        self.flats_plan_fn = flats_plan_fn
        self.start_time = start_time
        return
    
    def GetTargetInfo(self,ind):
        obj = self.schedule_table[ind]
        name = obj['Object_Name']
        mp = obj['Coordinates']
        pa = obj['PA']
        start = obj['Start_Datetime']
        end = obj['End_Datetime']
        filt = obj['Filter']
        dt = int((end - start).to('s').value)
        start = start.datetime.time().isoformat()
        return name,mp,pa,start,dt,filt
    
    def WriteTargetBlock(self,ind):
        name,mp,pa,start,dt,filt = self.GetTargetInfo(ind)
        ra = mp.ra
        dec = mp.dec
        block = r';===Target '+name+' ===\r'
        block += '#waituntil 1, '+start+'\r'
        block += '#repeat '+str(dt)+'\r'
        block += '#binning 1\r'+'#interval 1\r'+'#posang '+str(pa)+'\r'
        block += '#filter '+filt+'\r'
        block += name + '\t'+str(ra.deg)+'\t'+str(dec.deg)+'\r'
        return block
    
    def WriteDarkBlock(self,frames = 10):
        dark_start = self.start_time + 25*u.min
        dark_start = dark_start.datetime.time().isoformat()
        block = ';=== Target Dark ===\r'
        block += '#waituntil 1, '+dark_start+'\r'
        block += '#repeat 1\r'+'#count '+str(int(frames))+'\r'+'#interval 1\r'
        block += '#binning 1\r'+'#dark\r;\r'
        return block
    
    def WriteBiasBlock(self,frames = 10):
        bias_start = self.start_time + 30*u.min
        bias_start = bias_start.datetime.time().isoformat()
        block = ';=== Target Bias ===\r'
        block += '#waituntil 1, '+bias_start+'\r'
        block += '#repeat 1\r'+'#count '+str(int(frames))+'\r'+'#interval 0\r'
        block += '#binning 1\r'+'#bias\r;\r'
        return block

    def WriteFlatPlan(self,out_fn,flat_frames = 10):
        filter_names = np.unique(self.schedule_table['Filter'])
        block = ''
        self.flat_fn = out_fn
        for fn in filter_names:
            block +=str(int(flat_frames))+','+fn+',1,0.00\r'
        block+=';'
        f = open(out_fn,'w')
        f.write(block)
        f.close()
        return block
    
    def WritePlan(self,dark_frames = 10, bias_frames = 10,out_fn = 'test_plan.txt'):
        preamble = '#waituntil 1, '+self.start_time.datetime.time().isoformat()+'\r'
        preamble += '#chill -20'+'\r'
        preamble += '#duskflats '+self.flats_plan_fn+'\r'
        plan = preamble + self.WriteDarkBlock(dark_frames) + self.WriteBiasBlock(bias_frames)
        for i in range(len(self.schedule_table)):
            plan += self.WriteTargetBlock(i)
        plan += ';\r#shutdown\r;'
        f = open(out_fn,'w')
        f.write(plan)
        f.close()
        self.plan_fn = out_fn
        return plan
        

# #%%
# def UpdateConfigFile(fn,name):
#     return

# def LoadConfigFile(fn):
#     return

# #%%
# for i in obs.objects:
#     name = i.name
#     ra = i.coordinates[0].ra.hms
#     dec = DecDegToTriplet(i.coordinates[0])
#     print(name+': ')
#     print(ra,dec)
#     print('\n')
