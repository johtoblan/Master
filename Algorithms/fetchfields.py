import xarray as xr

def fetchfields(xarray,timeindex):

    geo_pl = xarray["geopotential_pl"].isel(time=timeindex)
    geo_sf = xarray["surface_geopotential"].isel(time=timeindex)
    airtemp_pl = xarray["air_temperature_pl"].isel(time=timeindex)
    rhs_pl = xarray["relative_humidity_pl"].isel(time=timeindex)
    lowcloud = xarray["low_type_cloud_area_fraction"].isel(time=timeindex)

    if timeindex > 0:
        prec = xarray["precipitation_amount_acc"].isel(time=timeindex) - xarray["precipitation_amount_acc"].isel(time=timeindex-1)
    else:
        prec = xarray["precipitation_amount_acc"].isel(time=timeindex)

    return  geo_pl, geo_sf, airtemp_pl, rhs_pl, lowcloud, prec
