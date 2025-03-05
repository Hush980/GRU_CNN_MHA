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

def compute_dis(forecast_time,fore_data,true_data):
	dis_sum=0
	dis_ds=[]
	for i in range(len(fore_data)):
		j=int(forecast_time/6)-1
		fore_lat=fore_data[i,j*2]
		fore_lon=fore_data[i,j*2+1]
		true_lat=true_data[i,j*2]
		true_lon=true_data[i,j*2+1]
		dis=getDistance(fore_lat, fore_lon, true_lat, true_lon)
		dis_sum+=dis
		dis_ds.append(dis)
	return dis_sum,dis_ds

fore_track=pd.read_csv('result_track_fore_nodrop.csv',sep=',',header=None)
fore_track=fore_track.values
true_track=pd.read_csv('result_track_truth_nodrop.csv',sep=',',header=None)
true_track=true_track.values
print(fore_track.shape)
dataframe=pd.read_csv('./IBTrACS_usa_2023.txt',sep=',',names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
i_index=[]
for i in range(len(dataframe)):
	if dataframe['name'][i]=='66666':
		i_index.append(i)

for i in range(len(dataframe)):
	if dataframe['lon'][i]<0:
		dataframe['lon'][i]=dataframe['lon'][i]+360
#2479

a=0
# drop_24=list(np.genfromtxt("./slp_latlon_2023_24.txt",dtype=int))
# drop_48=list(np.genfromtxt("./slp_latlon_2023_48.txt",dtype=int))
# drop_idx = sorted(list(set(drop_24 + drop_48)))
#
# j_index_all = []
# ii_index_all = []
# for i in range(len(i_index)):
# 	index=i_index[i]
# 	if i==len(i_index)-1:
# 		m=index+1
# 		n=len(dataframe)
# 	else:
# 		m=index+1
# 		n=i_index[i+1]
# 	i_str=str(i)
# 	if n - 12 > m + 4:
# 		for j in range(m + 4, n - 12):
# 			j_index_all.append(j)
# 			ii_index_all.append(i)
# drop_j=np.array(j_index_all)[drop_idx]
true_data=[]
fore_data=[]
for index_idx in range(len(i_index)):
	j_index=[]
	ii_index=[]
	for i in range(index_idx,index_idx+1):
		index=i_index[i]
		if i==len(i_index)-1:
			m=index+1
			n=len(dataframe)
		else:
			m=index+1
			n=i_index[i+1]
		i_str=str(i)
		if n - 12 > m + 4:
			for j in range(m + 4, n - 12):
				j_index.append(j)
				ii_index.append(i)
	# for drop in drop_j:
	# 	if drop in j_index:
	# 		j_index.remove(drop)
	b=a+len(j_index)
	if b>a:
		true_data.append(true_track[a:b])
		fore_data.append(fore_track[a:b])
	a=+b

true_lat=[]
true_lon=[]
for i in range(len(i_index)):
	index=i_index[i]
	if i==len(i_index)-1:
		m=index+1
		n=len(dataframe)
	else:
		m=index+1
		n=i_index[i+1]
	if n - 12 > m + 4:
		lat_data = dataframe['lat'][m:n].values
		lon_data = dataframe['lon'][m:n].values
		true_lat.append(lat_data)
		true_lon.append(lon_data)




def plot_space(p_i,h):
	lat_true = true_lat[h]
	lon_true = true_lon[h]
	lat1 = int(min(lat_true)) - 5
	lat2 = int(max(lat_true)) + 10
	lon1 = int(min(lon_true)) - 10
	lon2 = min(int(max(lon_true)) + 5,180)
	ax1 = fig.add_subplot(p_i,projection=ccrs.PlateCarree())
	ax1.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=1, zorder=1)
	extent=[lon1,lon2, lat1, lat2]
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
	lat_fore_24=fore_data[h][:,3*2]
	lon_fore_24=fore_data[h][:,3*2+1]
	lat_fore_48 = fore_data[h][:, 7 * 2]
	lon_fore_48 = fore_data[h][:, 7 * 2 + 1]
	lat_fore_72 = fore_data[h][:, 11 * 2]
	lon_fore_72 = fore_data[h][:, 11 * 2 + 1]

	for m in range(len(lat_true) - 1):
		pointA = list(lon_true)[m], list(lat_true)[m]
		pointB = list(lon_true)[m + 1], list(lat_true)[m + 1]
		geometries = ax1.add_geometries([sgeom.LineString([pointA, pointB])], crs=ccrs.PlateCarree(), color='b', lw=1)
	plt.plot(list(lon_true)[:], list(lat_true)[:], marker='.', color='b', label='observation value', lw=1,markersize=5)
	ax1.scatter(lon_true, lat_true, marker='.', color='b', s=15)
	for m in range(len(lat_fore_24) - 1):
		pointA = list(lon_fore_24)[m], list(lat_fore_24)[m]
		pointB = list(lon_fore_24)[m + 1], list(lat_fore_24)[m + 1]
		geometries = ax1.add_geometries([sgeom.LineString([pointA, pointB])], crs=ccrs.PlateCarree(), color='#F9C00F',lw=1)
	plt.plot(list(lon_fore_24), list(lat_fore_24), marker='.', color='#F9C00F', label='24-h prediction value', lw=1,markersize=5)
	ax1.scatter(lon_fore_24, lat_fore_24, marker='.', color='#F9C00F', s=15)
	for m in range(len(lat_fore_48) - 1):
		pointA = list(lon_fore_48)[m], list(lat_fore_48)[m]
		pointB = list(lon_fore_48)[m + 1], list(lat_fore_48)[m + 1]
		geometries = ax1.add_geometries([sgeom.LineString([pointA, pointB])], crs=ccrs.PlateCarree(), color='green',lw=1)
	plt.plot(list(lon_fore_48), list(lat_fore_48), marker='.', color='green', label='48-h prediction value', lw=1,markersize=5)
	ax1.scatter(lon_fore_48, lat_fore_48, marker='.', color='green', s=15)
	for m in range(len(lat_fore_72) - 1):
		pointA = list(lon_fore_72)[m], list(lat_fore_72)[m]
		pointB = list(lon_fore_72)[m + 1], list(lat_fore_72)[m + 1]
		geometries = ax1.add_geometries([sgeom.LineString([pointA, pointB])], crs=ccrs.PlateCarree(), color='r',lw=1)
	plt.plot(list(lon_fore_72), list(lat_fore_72), marker='.', color='r', label='72-h prediction value', lw=1,markersize=5)
	ax1.scatter(lon_fore_72, lat_fore_72, marker='.', color='red', s=15)
	plt.legend(loc='upper right', fontsize=8)
    # plt.colorbar(sc1,ax=ax1,orientation='horizontal',extend='max',fraction=0.05,pad=0.15,label='Distance (km)')

for h in range(5,6):
	fig = plt.figure(figsize=(5, 5), dpi=300)
	plt.rcParams.update({'font.family': 'Arial', 'axes.labelsize': 10, 'font.size': 10})
	plot_space(441,h)
	# plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.99, wspace=0.4,hspace=None)
	plt.savefig(f'./example_{h}.png')
	plt.show()