#!/usr/bin/env python
# coding: utf-8

# In[7]:


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import point
from glob import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import point
from glob import glob
from scipy.interpolate import griddata
import rasterio
from rasterio import plot
from osgeo import gdal 


# In[8]:


import pandas as pd
import numpy as np


# In[9]:


# this function conver the list of trips (csv files) imported to dataframes
# It format correctly the datetime 
def list_to_df (csv_list):
    dataframes_trip_number = []   
    for i in range(len(csv_list)):
        temp_df = pd.read_csv("./"+csv_list[i],skiprows=2, low_memory=False)
        dataframes_trip_number.append(temp_df)
        dataframes_trip_number[i]["Time(1)"] = pd.to_datetime(dataframes_trip_number[i]["Time(1)"], format="%d/%m/%Y %H:%M:%S.%f")
    return dataframes_trip_number


# In[10]:


#This function shows the statistics from a specific column name from all the trips available.
def compute_stat(list_, column):
    mean_values=[]
    for i in range(len(list_)):
        temp_mean=list_[i][column].mean()
        mean_values.append(temp_mean)
    mean_values_df = pd.DataFrame (mean_values, columns = [column + "_"+"mean"])
    max_values=[]
    for i in range(len(list_)):
        temp_max=list_[i][column].max()
        max_values.append(temp_max)
    max_values_df = pd.DataFrame (max_values, columns = [column + "_"+"max"])
    min_values=[]
    for i in range(len(list_)):
        temp_min=list_[i][column].min()
        min_values.append(temp_min)
    min_values_df = pd.DataFrame (min_values, columns = [column + "_"+"min"])
    median_values=[]
    for i in range(len(list_)):
        temp_median=list_[i][column].median()
        median_values.append(temp_median)
    median_values_df = pd.DataFrame (median_values, columns = [column + "_"+"median"])
    q25_values=[]
    for i in range(len(list_)):
        temp_q25=list_[i][column].quantile(0.25)
        q25_values.append(temp_q25)
    q25_values_df = pd.DataFrame (q25_values, columns = [column + "_"+"q25%"])
    final_table = mean_values_df.join([max_values_df,min_values_df,median_values_df,q25_values_df])
    q75_values=[]
    for i in range(len(list_)):
        temp_q75=list_[i][column].quantile(0.75)
        q75_values.append(temp_q75)
    q75_values_df = pd.DataFrame (q75_values, columns = [column + "_"+"q75%"]) 
    final_table = mean_values_df.join([max_values_df,min_values_df,median_values_df,q25_values_df, q75_values_df])
    
    return final_table


# In[11]:


#This functions read the survey from a csv. 
def import_survey(name_file):
    survey=pd.read_csv(name_file,header=None,sep=" ")
    survey.columns=["x","y","value"]
    survey_geo=gpd.GeoDataFrame(survey, geometry=gpd.points_from_xy(survey.x,survey.y))
    return survey_geo


# In[12]:


# Opening raster files and reading data
def openRaster(fn,access=0):
    ds=gdal.Open(fn)
    if ds is None:
        print("Error opening raster dataset")
    return ds

def getRasterBand(fn, band=1,access=0):
    ds=openRaster(fn, access)
    band=ds.GetRasterBand(band).ReadAsArray()
    return band


# In[13]:


#Creates a raster file (tiff) from the difference of two data. 
def createRasterFromcopy(fn, ds, data,driverFmt="Gtiff"):
    driver = gdal.GetDriverByName(driverFmt)
    outds = driver.CreateCopy(fn,ds,strict=0)
    outds.GetRasterBand(1).WriteArray(data)
    ds=None
    outds=None
def substractRasters(dat1,dat2,fnout):
    datout=dat2-dat1
    createRasterFromcopy(fnout,gdal.Open(Botlek_voor), datout)


# In[14]:


#this function use the extent of a raster to clip the raster.
def ClippingwithExtent(IMG1, IMG2):
    gt1=IMG1.GetGeoTransform()
    gt2=IMG2.GetGeoTransform()
    if gt1[0] < gt2[0]: #CONDITIONAL TO SELECT THE CORRECT ORIGIN
        gt3=gt2[0]
    else:
        gt3=gt1[0]
    if gt1[3] < gt2[3]:
        gt4=gt1[3]
    else:
        gt4=gt2[3]
    xOrigin = gt3
    yOrigin = gt4
    pixelWidth = gt1[1]
    pixelHeight = gt1[5]

    r1 = [gt1[0], gt1[3],gt1[0] + (gt1[1] * IMG1.RasterXSize), gt1[3] + (gt1[5] * IMG1.RasterYSize)]
    r2 = [gt2[0], gt2[3],gt2[0] + (gt2[1] * IMG2.RasterXSize), gt2[3] + (gt2[5] * IMG2.RasterYSize)]
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]

    xmin = intersection[0]
    xmax = intersection[2]
    ymin = intersection[3]
    ymax = intersection[1]

# Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1
    srs=IMG1.GetProjectionRef() #necessary to export with SRS

    img1 = IMG1.GetRasterBand(1).ReadAsArray(xoff, yoff, xcount, ycount)  
    img2 = IMG2.GetRasterBand(1).ReadAsArray(xoff, yoff, xcount, ycount)


# In[ ]:




