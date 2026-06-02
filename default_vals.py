#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from astropy import units as un, constants as const

default_cme_vals = {"dim":513,
                    "dR_factor":0.2, 
                    "vertical_height_factor":1, 
                    "horizontal_height_factor":0,
                    "depth_factor":1,
                    "mass":1e19*un.g,
                    "linear_extent": 6*un.au}

default_wind_vals = {"dim":513, 
                     "Mdot":6e-13*un.M_sun/un.yr, 
                     "vwind":400*un.km/un.s, 
                     "linear_extent":6*un.au}

default_star_vals = {"temp":5000*un.K, 
                     "radius":0.7*const.R_sun, 
                     "distance":3.2*un.pc, 
                     "dim":513, 
                     "linear_extent":6*un.au, 
                     "wavelength_range":[100*un.nm,1000*un.nm]}