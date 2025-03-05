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
import json
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

def compute_dis(fore_lat, fore_lon, true_lat, true_lon):
	dis_ds=[]
	for i in range(len(fore_lat)):
		dis=getDistance(fore_lat[i], fore_lon[i], true_lat[i], true_lon[i])
		dis_ds.append(dis)
	return dis_ds

def plot_space_track(p_i,lat_true,lon_true,lat_fore_24_xin,lon_fore_24_xin,lat_fore_24,lon_fore_24):
	lat1 = int(min(lat_true)) - 5
	lat2 = int(max(lat_true)) + 8
	lon1 = int(min(lon_true)) - 5
	lon2 = min(int(max(lon_true)) + 7,180)
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
	ax1.set_xticks(np.arange(extent[0]//5*5, extent[1]+5, 5))
	ax1.set_yticks(np.arange(extent[2]//5*5, extent[3]+5, 5))
	ax1.xaxis.set_major_formatter(LongitudeFormatter())
	# ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
	ax1.yaxis.set_major_formatter(LatitudeFormatter())
	# ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
	ax1.tick_params(axis='both', labelsize=8, direction='out')


	for m in range(len(lat_true) - 1):
		pointA = list(lon_true)[m], list(lat_true)[m]
		pointB = list(lon_true)[m + 1], list(lat_true)[m + 1]
		geometries = ax1.add_geometries([sgeom.LineString([pointA, pointB])], crs=ccrs.PlateCarree(), color='b', lw=1)
	plt.plot(list(lon_true)[:], list(lat_true)[:], marker='.', color='b', label='observation value', lw=1,markersize=5)
	ax1.scatter(lon_true, lat_true, marker='.', color='b', s=30)
	for m in range(len(lat_fore_24) - 1):
		pointA = list(lon_fore_24)[m], list(lat_fore_24)[m]
		pointB = list(lon_fore_24)[m + 1], list(lat_fore_24)[m + 1]
		geometries = ax1.add_geometries([sgeom.LineString([pointA, pointB])], crs=ccrs.PlateCarree(), color='#F9C00F',lw=1)
	plt.plot(list(lon_fore_24), list(lat_fore_24), marker='.', color='#F9C00F', label='24-h prediction value (GRU_CNN)', lw=1,markersize=5)
	ax1.scatter(lon_fore_24, lat_fore_24, marker='.', color='#F9C00F', s=30)
	for m in range(len(lat_fore_24_xin) - 1):
		pointA = list(lon_fore_24_xin)[m], list(lat_fore_24_xin)[m]
		pointB = list(lon_fore_24_xin)[m + 1], list(lat_fore_24_xin)[m + 1]
		geometries = ax1.add_geometries([sgeom.LineString([pointA, pointB])], crs=ccrs.PlateCarree(), color='green',lw=1)
	plt.plot(list(lon_fore_24_xin), list(lat_fore_24_xin), marker='.', color='green', label='24-h prediction value (GRU_CNN_MHA)', lw=1,markersize=5)
	ax1.scatter(lon_fore_24_xin, lat_fore_24_xin, marker='.', color='green', s=30)
	plt.legend(loc='upper right', fontsize=8)
    # plt.colorbar(sc1,ax=ax1,orientation='horizontal',extend='max',fraction=0.05,pad=0.15,label='Distance (km)')

def plot_space(p_i,vmax,dis_ds_24,lat_true,lon_true):
	lat1 = int(min(lat_true)) - 5
	lat2 = int(max(lat_true)) + 8
	lon1 = int(min(lon_true)) - 5
	lon2 = min(int(max(lon_true)) + 7,180)
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
	ax1.set_xticks(np.arange(extent[0]//5*5, extent[1]+5, 5))
	ax1.set_yticks(np.arange(extent[2]//5*5, extent[3]+5, 5))
	ax1.xaxis.set_major_formatter(LongitudeFormatter())
    # ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
	ax1.yaxis.set_major_formatter(LatitudeFormatter())
    # ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
	ax1.tick_params(axis='both', labelsize=8, direction='out')
	dis=dis_ds_24
	sc1=ax1.scatter(lon_true,lat_true,marker='o',c=dis,vmin=0,vmax=vmax,cmap='Reds',s=20)
	plt.colorbar(sc1,ax=ax1,orientation='horizontal',extend='max',fraction=0.040,pad=0.1,label='Distance (km)')

with open('../results/results_nodrop/fore_json.json','r',encoding='utf8')as fp:
	fore_track = json.load(fp)

with open('../results/results_nodrop/truth_json.json','r',encoding='utf8')as fp:
    true_track = json.load(fp)


dataframe=pd.read_csv('../typhoon10_24/IBTrACS_droptime_usa_2010_2024.txt',sep=',',names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
i_index=[]
for i in range(len(dataframe)):
	if dataframe['name'][i]=='66666':
		i_index.append(i)
for i in range(len(dataframe)):
	if dataframe['lon'][i]<0:
		dataframe['lon'][i]=dataframe['lon'][i]+360


def year_split(year, hours):
	i_index_name = []
	j_index_all = []
	for i in range(len(i_index)):
		index = i_index[i]
		if i == len(i_index) - 1:
			m = index + 1
			n = len(dataframe)
		else:
			m = index + 1
			n = i_index[i + 1]
		i_str = str(i)
		hh = int(hours / 6)
		j_index = []
		if str(dataframe['date'][m])[0:4] == str(year) and n - hh > m + 4:
			i_index_name.append(dataframe['name'][m])
			for j in range(m + 4, n - hh):
				j_index.append(j)
			j_index_all.append(j_index)
	return j_index_all, i_index_name

j_index_all_24=[]
j_index_all_48=[]
j_index_all_72=[]
i_index_name_24=[]
i_index_name_48=[]
i_index_name_72=[]
for year in range(2022,2025):
	j_index_all_24.append(year_split(year,24)[0])
	j_index_all_48.append(year_split(year,48)[0])
	j_index_all_72.append(year_split(year,72)[0])
	i_index_name_24.append(year_split(year,24)[1])
	i_index_name_48.append(year_split(year,48)[1])
	i_index_name_72.append(year_split(year,72)[1])



def split_fore(hours,j_index_all,prefix):

	str_h=str(hours)
	if prefix==1:
		lat_fore=fore_track[str_h]['lat']
		lon_fore=fore_track[str_h]['lon']
		lat_true=true_track[str_h]['lat']
		lon_true=true_track[str_h]['lon']
	else:
		dataframe_fore = pd.read_csv(f'F:/python_code/typhoon_track/results/results_nodrop/result_track_fore_{str_h}_gru.csv',names=['0', '1'])
		dataframe_truth = pd.read_csv(f'F:/python_code/typhoon_track/results/results_nodrop/result_track_truth_{str_h}_gru.csv',names=['0', '1'])
		lat_fore = dataframe_fore.values[:,0]
		lon_fore = dataframe_fore.values[:,1]
		lat_true = dataframe_truth.values[:,0]
		lon_true = dataframe_truth.values[:,1]
	m=0
	lat_fore_all=[]
	lon_fore_all=[]
	lat_true_all=[]
	lon_true_all=[]
	dis_ds_all=[]
	for j_index_all_y in j_index_all:
		lat_fore_ds=[]
		lon_fore_ds=[]
		lat_true_ds=[]
		lon_true_ds=[]
		dis_ds_ds=[]
		for kk in range(len(j_index_all_y)):
			n=m+len(j_index_all_y[kk])
			lat_fore_ds.append(lat_fore[m:n])
			lon_fore_ds.append(lon_fore[m:n])
			lat_true_ds.append(lat_true[m:n])
			lon_true_ds.append(lon_true[m:n])
			dis_ds=compute_dis(lat_fore[m:n], lon_fore[m:n], lat_true[m:n], lon_true[m:n])
			dis_ds_ds.append(dis_ds)
			m=n
		dis_ds_all.append(dis_ds_ds)
		lat_fore_all.append(lat_fore_ds)
		lon_fore_all.append(lon_fore_ds)
		lat_true_all.append(lat_true_ds)
		lon_true_all.append(lon_true_ds)
	return dis_ds_all,lat_fore_all,lon_fore_all,lat_true_all,lon_true_all


dis_ds_all_24,lat_fore_all_24,lon_fore_all_24,lat_true_all_24,lon_true_all_24=split_fore(24,j_index_all_24,0)
dis_ds_all_24_xin,lat_fore_all_24_xin,lon_fore_all_24_xin,_,_=split_fore(24,j_index_all_24,1)
for yy in range(0,3):
	i_ds_name_24=i_index_name_24[yy]
	# i_ds_name_48=i_index_name_48[yy]
	# i_ds_name_72=i_index_name_72[yy]
	# i_index_name_con = [i for i in i_ds_name_72 if i in i_ds_name_24 and i in i_ds_name_48]
	i_index_name_con = i_ds_name_24
	name_24=[]
	name_24_index = []
	for i in i_index_name_con:
		if i=='UNNAMED':
			# i_index_name_con.remove(i)
			print('unname')
		else:
			name_24_index.append(i_ds_name_24.index(i))
			name_24.append(i)
	lat_fore_ds_24=np.array(lat_fore_all_24[yy])[name_24_index]
	lon_fore_ds_24=np.array(lon_fore_all_24[yy])[name_24_index]

	lat_fore_ds_24_xin=np.array(lat_fore_all_24_xin[yy])[name_24_index]
	lon_fore_ds_24_xin = np.array(lon_fore_all_24_xin[yy])[name_24_index]

	lat_true_ds_24=np.array(lat_true_all_24[yy])[name_24_index]
	lon_true_ds_24=np.array(lon_true_all_24[yy])[name_24_index]
	dis_all_24 = np.array(dis_ds_all_24[yy])[name_24_index]
	dis_all_24_xin = np.array(dis_ds_all_24_xin[yy])[name_24_index]
	for kk,name in enumerate(name_24):
		lat_true_24=lat_true_ds_24[kk]
		lon_true_24=lon_true_ds_24[kk]
		lat_fore_24=lat_fore_ds_24[kk]
		lon_fore_24=lon_fore_ds_24[kk]

		dis_ds_24=dis_all_24[kk]
		dis_ds_24_xin=dis_all_24_xin[kk]
		lat_fore_24_xin=lat_fore_ds_24_xin[kk]
		lon_fore_24_xin=lon_fore_ds_24_xin[kk]
		fig = plt.figure(figsize=(4, 5), dpi=300)
		plt.rcParams.update({'font.family': 'Arial', 'axes.labelsize': 10, 'font.size': 10})
		plot_space_track(111,lat_true_24,lon_true_24,lat_fore_24_xin,lon_fore_24_xin,lat_fore_24,lon_fore_24)
		plt.savefig(f'./track1/{str(2022 + yy)}_{str(kk)}_{name}.png')
		# plot_space(111, 100, dis_ds_24_xin, lat_true_24, lon_true_24)
		# plt.savefig(f'./dis_24_xin/{str(2022 + yy)}_{str(kk)}_{name}.png')
		# plt.show()

