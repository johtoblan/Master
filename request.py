import urllib3
urllib3.disable_warnings()
import os
import cdsapi
import datetime
import subprocess

start_date = datetime.date(2008, 1, 1)
end_date = datetime.date(2019, 12, 31)
delta = datetime.timedelta(days=1)


c = cdsapi.Client()


while start_date <= end_date:
    year = start_date.year
    month = start_date.month
    day = start_date.day
    name = "source_disk/era5/general_patterns/%4.2i%2.2i%2.2i.nc"%(year,month,day)
    print(name)
    """    
    c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type" : "reanalysis",
                "format":"netcdf",
                "variable": [
                "vertical_velocity","temperature","geopotential"
                ],
                "pressure_level":[
                "975","950","925","900","875","850","825","800"
                ],
                "year": year,
                "month":month,
                "day": day,
                "time":list(range(24)),
                "area":
                [
                "75/-15/50/42"
                ]
            },
            name + "_tmp.nc")

    
    c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    'convective_precipitation', 'large_scale_precipitation', 'precipitation_type',
                ],
                "year": year,
                "month":month,
                "day": day,
                "time":list(range(24)),
                "area":
                [
                "75/-15/50/42"
                ]
            },
            name+"_tmp.nc")
    """
    #subprocess.run(["ncks","-C","-x","-v","t,z,w,level",name,name+"foo.nc"])
    subprocess.run(["ncks","-A",name+"_tmp.nc",name])
    #os.rename(name+"_tmp.nc",name)
    os.remove(name+"_tmp.nc")
    start_date +=delta
   
