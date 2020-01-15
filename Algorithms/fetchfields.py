import xarray as xr

def fetchfields(xarray,timeindex):
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
    variableperheight = (variable_pl.sel(pressure=850) - variable_pl.sel(pressure=925))/((geopotential_pl.sel(pressure=850) - geopotential_pl.sel(pressure=925))/9.81)
    result = variable_pl.sel(pressure=925) + variableperheight*(height -  geopotential_pl.sel(pressure=925)/9.81)
    return result
