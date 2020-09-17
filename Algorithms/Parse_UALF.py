#Need OrderedDict to represent the UALF (Universal ASCII Ligtning Format)
from collections import OrderedDict
import datetime

def parse_UALF(filename):
    """
    Takes in a filename of a file (ASCII-encoded) with lightning data in the UALF-format
    Returns a list of dictionaries, with a dictionary (ordered) for each line
    """
    #Creates a blank list to fill with dictionaries
    list_of_dictionaries = []


    keys = ["Version","Year","Month","Day of month","Hour","Minutes","Seconds","Nanoseconds","Latitude","Longitude","Peak Current",
            "Multiplicity", "Number of Sensors","Degrees of Freedom","Ellipse Angle","Semi-major Axis","Semi-minor Axis",
            "Chi-square Value","Rise Time", "Peak-to-zero Time", "Max Rate-of-Rise","Cloud Indicator","Angle Indicator",
            "Signal Indicator", "Timing Indicator"]

    with open(filename,"r") as infile:
        for line in infile:
            #Reads and splits the UALF-formatted line
            entries = line.split()

            #Creates a new OrderedDictionary
            dictionary = OrderedDict.fromkeys(keys)
            try:
                for number,key in enumerate(dictionary):
                    dictionary[key] = float(entries[number])
            except ValueError:
                print(line)
                continue
            except IndexError:
                print(line)
                continue

            #Make Date-Time values integer, to be easier to handle with pythons-datetime
            for key in ["Year","Month","Day of month","Hour","Minutes","Seconds","Nanoseconds"]:
                dictionary[key] = int(dictionary[key])

            list_of_dictionaries.append(dictionary)

    return list_of_dictionaries

def clean_UALF(dictionary):
    """
    Takes in a dictionary from the UALF-parser, and cleans it to a list with a time and place
    """

    time = datetime.datetime(dictionary["Year"],dictionary["Month"],dictionary["Day of month"],dictionary["Hour"], dictionary["Minutes"],dictionary["Seconds"],int(dictionary["Nanoseconds"]/1000))
    place = (dictionary["Latitude"],dictionary["Longitude"])
    return [time,place]

def get_Lyn_Data(filename):
    """
    Returns a list of lists of Date and Time, from parsing the UALF-file
    """
    list_of_date_and_time = []

    list_of_dictionaries = parse_UALF(filename)

    for entry in list_of_dictionaries:
        list_of_date_and_time.append(clean_UALF(entry))

    return list_of_date_and_time

def filter_UALF(inname,outname,filter):
    """
    filter is a list of minlat,maxlat,minlon,maxlon,
    anything outside this box is filtered out of outname
    """
    minlat,maxlat,minlon,maxlon = filter

    list_of_dictionaries = parse_UALF(inname)
    out_dictionary = []
    for entry in list_of_dictionaries:

        flag = 1
        if not (minlat < entry["Latitude"] and maxlat > entry["Latitude"]):
            flag = 0

        if not (minlon < entry["Longitude"] and maxlon > entry["Longitude"]):
            flag = 0

        if flag:
            out_dictionary.append(entry)

    return out_dictionary



if __name__ == '__main__':
    for dict in parse_UALF("../20170301flesland/lyndata01032017.dat"):
        print(clean_UALF(dict))
