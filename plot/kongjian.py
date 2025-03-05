import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter

import shapely.geometry as sgeom
def getDistance(latA, lonA, latB, lonB):
    ra = 6378136.49  # radius of equator: meter
    rb = 6356755  # radius of polar: meter

    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    if radLonA == radLonB:
        x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(0.001))
    else:
        x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr) / 1000
    return distance
fore_track=pd.read_csv('result_track_fore_nodrop.csv',sep=',',header=None)
fore_track=fore_track.values
true_track=pd.read_csv('result_track_truth_nodrop.csv',sep=',',header=None)
true_track=true_track.values
dis_ds_ds=[]
for i in range(len(fore_track)):
    dis_ds=[]
    for j in range(12):
        fore_lat = fore_track[i, j * 2]
        fore_lon = fore_track[i, j * 2 + 1]
        true_lat = true_track[i, j * 2]
        true_lon = true_track[i, j * 2 + 1]
        dis = getDistance(fore_lat, fore_lon, true_lat, true_lon)
        dis_ds.append(dis)
    dis_ds_ds.append(np.array(dis_ds))


# 设置渐变色
# clrs = []
# for i in np.linspace(16777215,16711680,10):
#     c = int(i)
#     clrs.append('#%06x'%c)
# colors =clrs
#
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
lon1=110
lon2=160
lat1=10
lat2=50
fig = plt.figure(figsize=(10, 3), dpi=300)
plt.rcParams.update({'font.family': 'Arial','axes.labelsize': 10, 'font.size':10})
def plot_space(p_i,i,vmax):
    ax1 = fig.add_subplot(p_i,projection=ccrs.PlateCarree())
    ax1.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=1, zorder=1)
    extent=[lon1, lon2, lat1, lat2]
    ax1.set_extent(extent,crs=ccrs.PlateCarree())
    ax1.add_feature(cfeat.OCEAN, edgecolor='black',facecolor='None')
    ax1.add_feature(cfeat.LAND, edgecolor='black', facecolor='None')
    ax1.add_feature(cfeat.LAKES, edgecolor='black', facecolor='None')
    ax1.add_feature(cfeat.RIVERS,  edgecolor='black',facecolor='None')
    gl1 = ax1.gridlines(draw_labels=False, linewidth=0.1, color='k', alpha=0.5, linestyle='--')
    gl1.xlabels_top = gl1.ylabels_right = False
    ax1.set_xlim(lon1,lon2)
    ax1.set_ylim(lat1,lat2)
    ax1.set_xticks(np.arange(extent[0]//10*10, extent[1]+10, 10))
    ax1.set_yticks(np.arange(extent[2]//10*10, extent[3]+10, 10))
    ax1.xaxis.set_major_formatter(LongitudeFormatter())
    # ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    # ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.tick_params(axis='both', labelsize=8, direction='out')
    lat_true=true_track[:,i*2]
    lon_true=true_track[:,i*2+1]
    dis=np.array(dis_ds_ds)[:,i]
    print(max(dis))
    sc1=ax1.scatter(lon_true,lat_true,marker='.',c=dis,vmin=0,vmax=vmax,cmap='Reds',s=10)
    plt.colorbar(sc1,ax=ax1,orientation='horizontal',extend='max',fraction=0.05,pad=0.15,label='Distance (km)')


plot_space(131,3,100)
plot_space(132,7,200)
plot_space(133,11,400)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.99, wspace=0.4,hspace=None)
plt.savefig('./space.png')
plt.show()