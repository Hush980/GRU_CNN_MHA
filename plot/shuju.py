#!/student/home/zsq/wangliang/anaconda3/envs/pytorch/bin/python3.9
#encoding:utf-8
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import random
import pandas as pd
import datetime
import xarray as xr
from mpl_toolkits.basemap import Basemap
dataframe=pd.read_csv('IBTrACS_droptime.txt',sep=',',names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
print(dataframe[0:20])
i_index=[]
for i in range(len(dataframe)):
    if dataframe['name'][i]=='66666':
        i_index.append(i)

j_index=[]
ii_index=[]
for i in range(1865,len(i_index)):
    index=i_index[i]
    if i==len(i_index)-1:
        m=index+1
        n=len(dataframe)
    else:
        m=index+1
        n=i_index[i+1]
    i_str=str(i)
    if n-24>m+8:
        for j in range(m+8,n-24):
            j_index.append(j)
            ii_index.append(i)

datetime_list=dataframe['date']
j_names=j_index
i_names=ii_index

slp_label_ds=[]
idx=200
j = j_names[idx]
i = i_names[idx]
print(i,j)
slp_ds = []
lat=dataframe['lat'][j]
lon = dataframe['lon'][j]
for k in range(j+6, j+8,2):
    i_str=str(i)
    DATE=datetime.datetime.strptime(datetime_list[k][0:19],'%Y-%m-%d %H:%M:%S')
    print(DATE)
    datetime_str=DATE.strftime('%Y%m%d%H')
    nc_fp = f'D:/nc_data/slp_{datetime_str[0:4]}_{datetime_str[4:6]}.nc'
    nc_data = xr.open_dataset(nc_fp)
    datetime_utc = datetime.datetime.strptime(datetime_str, "%Y%m%d%H")
    datetime_utc = datetime_utc.strftime("%Y-%m-%dT%H:%M:%S")
    SLP = nc_data['msl'].sel(time=datetime_utc)
    longitude_ds = nc_data.longitude
    latitude_ds = nc_data.latitude
    lon_nearest = nc_data.longitude.sel(longitude=lon, method='nearest').values
    lat_nearest = nc_data.latitude.sel(latitude=lat, method='nearest').values
    print(lat_nearest, lon_nearest)
    lat0 = lat_nearest - 20
    lat1 = lat_nearest + 20
    lon0 = lon_nearest - 20
    lon1 = lon_nearest + 20
    slp = SLP.loc[lat1:lat0:2, lon0:lon1:2].values

    LON = nc_data.longitude.loc[lon0:lon1:2].values
    LAT = nc_data.latitude.loc[lat1:lat0:2].values
    grid_lon, grid_lat = np.meshgrid(LON, LAT)
    flat_lon = grid_lon.flatten()  # 将坐标展成一维
    flat_lat = grid_lat.flatten()
    flat_points = np.column_stack((flat_lon, flat_lat))
    flat_points = flat_points.reshape(81, 81, 2)
    slp_lat=flat_points[:,:,1]
    slp_lon = flat_points[:, :, 0]

    fig = plt.figure(figsize=(7, 7), dpi=400)  # 画布尺寸
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
    plt.rcParams['axes.unicode_minus'] = False
    m = Basemap(projection='cyl', ax=ax,  # 投影类型
                llcrnrlon=lon0, llcrnrlat=lat0,  # 左下角经纬的
                urcrnrlon=lon1, urcrnrlat=lat1,  # 右上角经纬度
                )
    m.drawcoastlines()  # 海岸线
    m.drawcountries()  # 国界线
    parallels = np.arange(lat0, lat1+5, 5)
    m.drawparallels(parallels, ax=ax, labels=[1, 0, 0, 0], dashes=[1,400],fontsize=12,linewidth=0.01)  # 绘制纬线

    meridians = np.arange(lon0, lon1 + 5, 5)
    m.drawmeridians(meridians, ax=ax, labels=[0, 0, 0, 1],  dashes=[1,400],fontsize=12, linewidth=0.01)  # 绘制经线
    wlon = list(np.linspace(lon0, lon1, 81))
    wlat = list(np.linspace(lat1, lat0, 81))
    lon_ds, lat_ds = np.meshgrid(wlon, wlat)
    xi, yi = m(lon_ds, lat_ds)
    max_z=int((max(map(max, slp))//4+1)*4)
    min_z = int((min(map(min, slp))//4-1)*4)
    levels=np.arange(int(min_z), int(max_z+1), 100)
    cs = m.contourf(xi, yi,slp,levels,cmap='jet',extend='both')
    C = m.contour(xi, yi,slp,levels, colors="black", linewidths=1.2,linestyles='-')
    # cbar = m.colorbar(cs, location='bottom', pad="13%",extend='max')
    # plt.clabel(C, inline=True, fontsize=10,fmt='%.0f')
    # plt.savefig('./plot/example_18.png', transparent=True)
    # plt.show()
