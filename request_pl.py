import urllib3
urllib3.disable_warnings()
import os
import cdsapi
import pandas as pd

dds = pd.read_csv("source/dataset_new.csv")
ands = pd.read_csv("source/analysis_dataset_new.csv")
c = cdsapi.Client()

for ds in [dds,ands]:
    for i, row in ds.iterrows():
        date = str(row[0])
        year = int(date[0:4])
        month = int(date[4:6])
        day = int(date[6:8])
        time = int(row[1])
        print(time)
        name = "source_disk/era5/specific_patterns/%4.2i%2.2i%2.2i%2.2i.nc"%(year,month,day,time)
        try:
            # Checks if a file exists, if error, downloads a file.
            open(name+"_tmp")
            continue
        except:
            pass
        print(name)
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type" : "reanalysis",
                "format":"netcdf",
                'variable':[
                    'u_component_of_wind', 'v_component_of_wind',
                ],
                'pressure_level':[
                    '1000',"975","950","925","900","875","850","825","800","775","750","700","650","600","550","500","450","400","350","300","250","225","200"
                ],
                "year": year,
                "month":month,
                "day": day,
                "time":time,
                "area":
                [
                "75/-15/50/42"
                ]
            },
            name+"_tmp")
