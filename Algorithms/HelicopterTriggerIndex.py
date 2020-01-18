from math import ceil,floor
import xarray as xr
import numpy as np
import numba

class HelicopterTriggerIndex(object):
    """docstring for HelicopterTriggerIndex."""

    def __init__(self, functions, neighbourhood_size = 1):
        self.functions = functions
        self.N = len(functions)
        self.nSize = neighbourhood_size

    def __call__(self, arrays):
        """
        arrays is a list of arrays, self.N long, which are either 1x1 or neighbourhood_size X neighbourhood_size big

        return HelicopterTriggerIndex with values in [0,1]
        """

        for i,subfunction in enumerate(self.functions):
            if i == 0:
                HTI = subfunction(arrays[i])
            else:
                HTI += subfunction(arrays[i])

        # Average subIndexes to last HTI
        return HTI/self.N

def temperature_max_band_from_b_to_c(b,c):
    """
    NOTE: C < B
    Gives high risk (1 or 100%) for temperatures in [c,b] gives [0,1] for [a,b]
    and [1,0] for [c,d] (linear mapping)
    """
    # Define a and d for linear mapping
    a = b + 1
    d = c - 1
    def f(T):
        a_slice = np.where(T > -1,1,0)
        a_slice[np.where(T > 0)] = 0
        b_slice = np.where(T < -1,1,0)
        c_slice = np.where(T > -6,1,0)

        d_slice = np.where(T > -7,1,0)
        d_slice[np.where(T > -6)] = 0

        new_slice = np.where(b_slice == c_slice,1.0,0.0)
        new_slice[np.where(a_slice)] = abs(abs(T[np.where(a_slice)] -abs(a)))
        new_slice[np.where(d_slice)] =  abs(abs(T[np.where(d_slice)])  - abs(d))
        return new_slice
    return f

@numba.jit
def neighbourhood_max(xarrayvalues,neigbhoursize = 3):
    N = neigbhoursize
    # Assume incoming object to be numpy array
    X,Y = xarrayvalues.shape
    result = np.zeros((X,Y))
    # First fill non-trivial parts
    for y in range(N,Y-N + 1+1):
        for x in range(N,X-N + 1+1):
            result[x,y] = np.max(xarrayvalues[x-N:x+N + 1,y-N:y+N + 1])
    for i in range(N):
        for y in range(Y):
            result[i,y] = xarrayvalues[i,y]
            result[-(1+i),y] = xarrayvalues[-(1+i),y]
        for x in range(X):
            result[x,i] = xarrayvalues[x,i]
            result[x,-(1+i)] = xarrayvalues[x,-(1+i)]
    return result

@numba.jit
def neighbourhood_min(xarrayvalues,neigbhoursize = 3):
    N = neigbhoursize
    # Assume incoming object to be numpy array
    X,Y = xarrayvalues.shape
    result = np.zeros((X,Y))
    # First fill non-trivial parts
    for y in range(N,Y-N + 1+1):
        for x in range(N,X-N + 1+1):
            result[x,y] = np.min(xarrayvalues[x-N:x+N + 1,y-N:y+N + 1])
    for i in range(N):
        for y in range(Y):
            result[i,y] = xarrayvalues[i,y]
            result[-(1+i),y] = xarrayvalues[-(1+i),y]
        for x in range(X):
            result[x,i] = xarrayvalues[x,i]
            result[x,-(1+i)] = xarrayvalues[x,-(1+i)]
    return result

def only_positive_but_no_larger_than_1(array):
    size = array.shape
    result = np.zeros(size)
    result[np.where(array > 0)] = np.minimum(1.5*array[np.where(array>0)],1)
    return result

def only_positive_but_no_larger_than_1_with_scaling(topoarray,scaling = 15000):
    def f(array):
        size = array.shape
        result = np.zeros(size)
        result[np.where(array > 0) and np.where(topoarray < 10)] = np.minimum(1.5*array[np.where(array>0) and np.where(topoarray < 10)],1)
        scalingfactor = 1-(topoarray[np.where(array>0) and np.where(topoarray>10)]/scaling)
        result[np.where(array > 0) and np.where(topoarray > 10)] = np.minimum(scalingfactor*1.5*array[np.where(array > 0) and np.where(topoarray > 10)],1)
        return result
    return f

def fetchfields(xarray,timeindex):
    # Fetch fields used for operational HTI - post proccessing
    geo_pl = xarray["geopotential_pl"].isel(time=timeindex)
    geo_sf = xarray["surface_geopotential"].isel(time=timeindex)
    airtemp_pl = xarray["air_temperature_pl"].isel(time=timeindex)
    rhs_pl = xarray["relative_humidity_pl"].isel(time=timeindex)
    upward_pl = xarray["upward_air_velocity_pl"].isel(time=timeindex)

    lowcloud = xarray["low_type_cloud_area_fraction"].isel(time=timeindex)
    if timeindex > 0:
        prec = xarray["precipitation_amount_acc"].isel(time=timeindex) - xarray["precipitation_amount_acc"].isel(time=timeindex-1)
    else:
        prec = xarray["precipitation_amount_acc"].isel(time=timeindex)

    return  geo_pl, geo_sf, airtemp_pl, rhs_pl, upward_pl, lowcloud, prec

def get_height_value_from_pl(geopotential_pl,variable_pl,height=750):
    # Assume 925 to 850 band always has interesting heights. May need changing later
    # Also Assume 9.81 is correct gravitational constant ( should be to 1%)
    z2,z1 = geopotential_pl.sel(pressure=850)/9.81 , geopotential_pl.sel(pressure=925)/9.81
    v2,v1 = variable_pl.sel(pressure=850) , variable_pl.sel(pressure=925)
    variableperheight = (v2-v1)/(z2-z1)
    result = v1 + variableperheight*(height - z1)
    return result

if __name__ == '__main__':
    ds = xr.open_dataset("http://thredds.met.no/thredds/dodsC/meps25epsarchive/2019/12/09/meps_extracted_2_5km_20191209T06Z.nc")
    # 3 is the third timestep
    geo_pl, geo_sf, airtemp_pl, rhs_pl, upward_pl, lowcloud, prec = fetchfields(ds,3)
    # convert to C from Kelvin
    air_temp = get_height_value_from_pl(geo_pl,airtemp_pl-273.15)
    W_ = get_height_value_from_pl(geo_pl,upward_pl)

    W = np.zeros_like(W_)
    topomin = np.zeros_like(W)
    topomax = np.zeros_like(topomin)
    cl = np.zeros_like(lowcloud.values[0,...])
    pc = np.zeros_like(prec.values[0,...])


    Nbh = 14
    for member in W_["ensemble_member"]:
        # Find area of highest convection in the neighbourhood
        W[member] = neighbourhood_max(W_.isel(ensemble_member=member).values,Nbh)
        # We want to avoid stratus clouds.
        cl[member] = neighbourhood_max(lowcloud.isel(ensemble_member=member).values[0,...],Nbh)
        cl[member] -= neighbourhood_min(lowcloud.isel(ensemble_member=member).values[0,...],Nbh)

        pc[member] = neighbourhood_max(prec.isel(ensemble_member=member).values[0,...],Nbh)

        # Calculate maximum and minimum topographies
        topomax[member] = neighbourhood_max(geo_sf.isel(ensemble_member=member).values[0,...],Nbh)
        topomin[member] = neighbourhood_min(geo_sf.isel(ensemble_member=member).values[0,...],Nbh)

    # Remove part where convection may be negative
    W[np.where(W < 0)] = 0

    # Define functions
    tfunc = temperature_max_band_from_b_to_c(-1,-6)
    wfunc = only_positive_but_no_larger_than_1#_with_scaling(topomax)
    pfunc = only_positive_but_no_larger_than_1
    cfunc = lambda x: x

    HTI = HelicopterTriggerIndex([tempfunc,wfunc,pfunc,cfunc])
