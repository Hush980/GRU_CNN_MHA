import pandas as pd
import numpy as np
import math
import json
from math import radians, cos, sin, sqrt, atan2
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径，单位为千米
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    distance = 2 * R * atan2(sqrt(a), sqrt(1-a))

    bearing = atan2(sin(lon2-lon1)*cos(lat2), cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1))
    bearing = (bearing + 360) % 360
    return bearing
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


def compute_dis(hours,fore_data,true_data):
	dis_ds=[]
	j=str(hours)
	for i in range(len(fore_data[j]['lat'])):
		fore_lat=fore_data[j]['lat'][i]
		fore_lon=fore_data[j]['lon'][i]
		true_lat=true_data[j]['lat'][i]
		true_lon=true_data[j]['lon'][i]
		dis=getDistance(fore_lat, fore_lon, true_lat, true_lon)
		dis_ds.append(dis)
	return dis_ds

#距离稳定度
def compute_dis_stablity(dis_ds_24,number):
	i_sum=0
	for i in dis_ds_24:
		if i<=number:
			i_sum+=1
	return i_sum/len(dis_ds_24)


with open('../results/results_drop/fore_json.json','r',encoding='utf8')as fp:
	fore_track = json.load(fp)

with open('../results/results_drop/truth_json.json','r',encoding='utf8')as fp:
    true_track = json.load(fp)

dis_ds_24=compute_dis(24,fore_track,true_track)
dis_ds_48=compute_dis(48,fore_track,true_track)
dis_ds_72=compute_dis(72,fore_track,true_track)

dis_stablity_24=compute_dis_stablity(dis_ds_24,100)
dis_stablity_48=compute_dis_stablity(dis_ds_48,200)
dis_stablity_72=compute_dis_stablity(dis_ds_72,300)


dataframe=pd.read_csv('../typhoon10_24/IBTrACS_droptime_usa_2010_2024.txt',sep=',',names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
i_index=[]
for i in range(len(dataframe)):
	if dataframe['name'][i]=='66666':
		i_index.append(i)
print(len(i_index))
for i in range(len(dataframe)):
	if dataframe['lon'][i]<0:
		dataframe['lon'][i]=dataframe['lon'][i]+360


def j_index_com(hours,n1):
	drop_idx = list(np.genfromtxt(f"../typhoon10_24/drop_idx_{str(hours)}.txt", dtype=int))
	j_index0 = []
	ii_index = []
	for i in range(len(i_index)):
		index=i_index[i]
		if i==len(i_index)-1:
			m=index+1
			n=len(dataframe)
		else:
			m=index+1
			n=i_index[i+1]
		i_str=str(i)
		hh=int(hours/6)
		if n - hh > m + 4:
			for j in range(m + 4, n - hh):
				j_index0.append(j)
				ii_index.append(i)
	j_index=np.delete(j_index0,drop_idx,0)
	j_index_new=j_index[n1:]
	return j_index_new


# #方向稳定度（PS）
def compute_dis(j_index,hours,fore_data,true_data):
	i_sum=0
	j=str(hours)
	for i in range(len(fore_data[j]['lat'])):
		lat0=dataframe['lat'][j_index[i]]
		lon0=dataframe['lon'][j_index[i]]
		fore_lat=fore_data[j]['lat'][i]
		fore_lon=fore_data[j]['lon'][i]
		true_lat=true_data[j]['lat'][i]
		true_lon=true_data[j]['lon'][i]
		lat_dt_f=lat0-fore_lat
		lon_dt_f=lon0-fore_lon
		lat_dt_t=lat0-true_lat
		lon_dt_t=lon0-true_lon
		lat_s=lat_dt_f*lat_dt_t
		lon_s=lon_dt_f*lon_dt_t
		if lat_s>0 and lon_s>0:
			i_sum+=1
	return i_sum/len(fore_data[j]['lat'])

j_index_new_24=j_index_com(24,6205-433)
j_index_new_48=j_index_com(48,5093-439)
j_index_new_72=j_index_com(72,4080-393)

direct_stablity_24=compute_dis(j_index_new_24,24,fore_track,true_track)
direct_stablity_48=compute_dis(j_index_new_48,48,fore_track,true_track)
direct_stablity_72=compute_dis(j_index_new_72,72,fore_track,true_track)
print(direct_stablity_24)
#有效稳定度
def compute_dis(j_index,hours,fore_data,true_data,number):
	i_sum=0
	j=str(hours)
	for i in range(len(fore_data[j]['lat'])):
		lat0=dataframe['lat'][j_index[i]]
		lon0=dataframe['lon'][j_index[i]]
		fore_lat=fore_data[j]['lat'][i]
		fore_lon=fore_data[j]['lon'][i]
		true_lat=true_data[j]['lat'][i]
		true_lon=true_data[j]['lon'][i]
		lat_dt_f=lat0-fore_lat
		lon_dt_f=lon0-fore_lon
		lat_dt_t=lat0-true_lat
		lon_dt_t=lon0-true_lon
		lat_s=lat_dt_f*lat_dt_t
		lon_s=lon_dt_f*lon_dt_t
		dis=getDistance(fore_lat, fore_lon, true_lat, true_lon)
		if lat_s>0 and lon_s>0 and dis<number:
			i_sum+=1
	return i_sum/len(fore_data[j]['lat'])

effect_stablity_24=compute_dis(j_index_new_24,24,fore_track,true_track,100)
effect_stablity_48=compute_dis(j_index_new_48,48,fore_track,true_track,200)
effect_stablity_72=compute_dis(j_index_new_72,72,fore_track,true_track,300)

#转向灵敏度
def compute_turning(j_index,hours,fore_data,true_data):
	i_all_sum=0
	i_sum=0
	j=str(hours)
	for i in range(len(fore_data[j]['lat'])):
		lat0=dataframe['lat'][j_index[i]]
		lon0=dataframe['lon'][j_index[i]]
		lat0_24=dataframe['lat'][j_index[i]-4]
		lon0_24=dataframe['lon'][j_index[i]-4]

		fore_lat=fore_data[j]['lat'][i]
		fore_lon=fore_data[j]['lon'][i]
		true_lat=true_data[j]['lat'][i]
		true_lon=true_data[j]['lon'][i]
		lat_dt_24=lat0_24-lat0
		lon_dt_24=lon0_24-lon0

		lat_dt_f=lat0-fore_lat
		lon_dt_f=lon0-fore_lon
		lat_dt_t=lat0-true_lat
		lon_dt_t=lon0-true_lon

		lat_t=lat_dt_24*lat_dt_t
		lon_t=lon_dt_24*lon_dt_t
		lat_f=lat_dt_24*lat_dt_f
		lon_f=lon_dt_24*lon_dt_f
		brng0_t=haversine(lat0_24, lon0_24, lat0, lon0)
		brng1_t=haversine(lat0, lon0, true_lat, true_lon)
		brng0_f=haversine(lat0_24, lon0_24, lat0, lon0)
		brng1_f=haversine(lat0, lon0, fore_lat, fore_lon)
		if lat_t>0 and lon_t>0 and abs(brng1_t-brng0_t)<110:
			i_all_sum+=1
		if lat_t<0 or lon_t<0:
			i_all_sum+=1

		if lat_t>0 and lon_t>0 and abs(brng1_t-brng0_t)<110 and lat_f>0 and lon_f>0 and abs(brng1_f-brng0_f)<110:
			i_sum+=1
		if lat_t<0 or lon_t<0 and lat_f<0 or lon_f<0:
			i_sum+=1
	return i_sum/i_all_sum

turning_stablity_24=compute_turning(j_index_new_24,24,fore_track,true_track)
turning_stablity_48=compute_turning(j_index_new_48,48,fore_track,true_track)
turning_stablity_72=compute_turning(j_index_new_72,72,fore_track,true_track)
print(turning_stablity_72)

#变速灵敏度
def ave_speed(lat_ds,lon_ds):
	dis_ds=[]
	for i in range(len(lat_ds)-1):
		lat0=lat_ds[i]
		lon0=lon_ds[i]
		lat1=lat_ds[i+1]
		lon1=lon_ds[i+1]
		dis=getDistance(lat0, lon0, lat1, lon1)
		dis_ds.append(dis)
	return np.mean(dis_ds)



j_index_new_6 = j_index_com(6, 7063 - 437)
j_index_new_12 = j_index_com(12, 6775 - 408)
j_index_new_18 = j_index_com(18, 6489 - 427)
j_index_new_24 = j_index_com(24, 6205 - 433)
j_index_new_30 = j_index_com(30, 5922 - 433)
j_index_new_36 = j_index_com(36, 5643 - 442)
j_index_new_42 = j_index_com(42, 5366 - 459)
j_index_new_48 = j_index_com(48, 5093 - 439)
j_index_new_54 = j_index_com(54, 4828 - 437)
j_index_new_60 = j_index_com(60, 4568 - 408)
j_index_new_66 = j_index_com(66, 4318 - 393)
j_index_new_72 = j_index_com(72, 4080 - 393)

c2 = [c_i for c_i in j_index_new_72 if
	  c_i in j_index_new_6 and c_i in j_index_new_12 and c_i in j_index_new_18 and c_i in j_index_new_24 and c_i in j_index_new_30
	  and c_i in j_index_new_36 and c_i in j_index_new_42 and c_i in j_index_new_48 and c_i in j_index_new_54 and c_i in j_index_new_60 and c_i in j_index_new_66]


def recom_latlon(hours, j_index_new,true_track):
	ii = str(hours)
	index = [list(j_index_new).index(i) for i in c2]
	lat_ds = true_track[ii]['lat']
	lon_ds = true_track[ii]['lon']
	lat_true_new = np.array(lat_ds)[index]
	lon_true_new = np.array(lon_ds)[index]
	return lat_true_new, lon_true_new



def compute_speed(j_index_new_con,true_track):
	lat_true_new_ds = []
	lon_true_new_ds = []
	for m in range(0, 12):
		hours = (m + 1) * 6
		str_bianl = f'j_index_new_{str(hours)}'
		bianl = globals()[str_bianl]
		lat_true_new, lon_true_new = recom_latlon(hours, bianl, true_track)
		lat_true_new_ds.append(lat_true_new)
		lon_true_new_ds.append(lon_true_new)

	dis_24_0_ds=[]
	dis0_24_ds=[]
	dis24_48_ds=[]
	dis48_72_ds=[]
	for i in range(len(j_index_new_con)):
		lat0=dataframe['lat'][j_index_new_con[i]]
		lon0=dataframe['lon'][j_index_new_con[i]]
		lat0_ds=[]
		lon0_ds=[]
		for m in range(5):
			lat0_24=dataframe['lat'][j_index_new_con[i]-m]
			lon0_24=dataframe['lon'][j_index_new_con[i]-m]
			lat0_ds.append(lat0_24)
			lon0_ds.append(lon0_24)
		dis_24_0=ave_speed(lat0_ds,lon0_ds)


		lat1_ds_t=[]
		lon1_ds_t=[]
		lat1_ds_t.append(lat0)
		lon1_ds_t.append(lon0)

		for m in range(0,4):
			lat_true_new=lat_true_new_ds[m]
			lon_true_new = lon_true_new_ds[m]
			true_lat=lat_true_new[i]
			true_lon=lon_true_new[i]
			lat1_ds_t.append(true_lat)
			lon1_ds_t.append(true_lon)
		dis0_24=ave_speed(lat1_ds_t,lon1_ds_t)


		lat2_ds_t=[]
		lon2_ds_t=[]
		for m in range(3,8):
			lat_true_new=lat_true_new_ds[m]
			lon_true_new = lon_true_new_ds[m]
			true_lat=lat_true_new[i]
			true_lon=lon_true_new[i]
			lat2_ds_t.append(true_lat)
			lon2_ds_t.append(true_lon)

		dis24_48=ave_speed(lat2_ds_t,lon2_ds_t)

		lat3_ds_t=[]
		lon3_ds_t=[]
		for m in range(7,12):
			lat_true_new=lat_true_new_ds[m]
			lon_true_new = lon_true_new_ds[m]
			true_lat=lat_true_new[i]
			true_lon=lon_true_new[i]
			lat3_ds_t.append(true_lat)
			lon3_ds_t.append(true_lon)

		dis48_72=ave_speed(lat3_ds_t,lon3_ds_t)

		dis_24_0_ds.append(dis_24_0)
		dis0_24_ds.append(dis0_24)
		dis24_48_ds.append(dis24_48)
		dis48_72_ds.append(dis48_72)
	return dis_24_0_ds,dis0_24_ds,dis24_48_ds,dis48_72_ds

dis_24_0_ds_t,dis0_24_ds_t,dis24_48_ds_t,dis48_72_ds_t=compute_speed(c2,true_track)
dis_24_0_ds_f,dis0_24_ds_f,dis24_48_ds_f,dis48_72_ds_f=compute_speed(c2,fore_track)

def compute_speed_change(dis0_24_ds_t,dis0_24_ds_f,dis_24_0_ds_t,dis_24_0_ds_f):
	i_all_sum=0
	i_sum=0
	for i in range(len(dis_24_0_ds_t)):
		if dis0_24_ds_t[i]/dis_24_0_ds_t[i]>2:
			i_all_sum+=1
		if dis0_24_ds_t[i]/dis_24_0_ds_t[i]>2 and dis0_24_ds_f[i]/dis_24_0_ds_f[i]>2:
			i_sum+=1

		if dis0_24_ds_t[i]/dis_24_0_ds_t[i]<0.5:
			i_all_sum+=1
		if dis0_24_ds_t[i]/dis_24_0_ds_t[i]<0.5 and dis0_24_ds_f[i]/dis_24_0_ds_f[i]<0.5:
			i_sum+=1
	return i_sum/i_all_sum

speed_stablity_24=compute_speed_change(dis0_24_ds_t,dis0_24_ds_f,dis_24_0_ds_t,dis_24_0_ds_f)
speed_stablity_48=compute_speed_change(dis24_48_ds_t,dis24_48_ds_f,dis0_24_ds_t,dis0_24_ds_f)
speed_stablity_72=compute_speed_change(dis48_72_ds_t,dis48_72_ds_f,dis24_48_ds_t,dis24_48_ds_f)
print(speed_stablity_72)

#plot
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Arial', 'axes.labelsize': 10, 'font.size': 10})
# plt.rcParams['font.family'] = 'Microsoft YaHei'
labels=np.array(["Distance\nStability","Path follow\nstability","Effective\nStability","Turning\nsensitivity","Variable speed\nsensitivity"])
# labels=np.array(["距离\n稳定度","方向\n稳定度","有效\n稳定度","转向\n灵敏度","变速\n灵敏度"])
stats_24=[dis_stablity_24,direct_stablity_24,effect_stablity_24,turning_stablity_24,speed_stablity_24]
stats_48=[dis_stablity_48,direct_stablity_48,effect_stablity_48,turning_stablity_48,speed_stablity_48]
stats_72=[dis_stablity_72,direct_stablity_72,effect_stablity_72,turning_stablity_72,speed_stablity_72]
# 画图数据准备，⻆度、状态值
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
stats_24=np.concatenate((stats_24,[stats_24[0]]))
stats_48=np.concatenate((stats_48,[stats_48[0]]))
stats_72=np.concatenate((stats_72,[stats_72[0]]))
angles=np.concatenate((angles,[angles[0]]))
labels=np.concatenate((labels,[labels[0]]))
# ⽤Matplotlib画蜘蛛图
fig = plt.figure(figsize=(18,6))
ax1 = fig.add_subplot(131, polar=True)
line1,=ax1.plot(angles, stats_24, 'o-', linewidth=2,label='24h') # 连线
ax1.fill(angles, stats_24, alpha=0.25) # 填充
ax1.set_thetagrids(angles*180/np.pi,labels,fontsize = 18)
ax1.set_rgrids([0.2,0.4,0.6,0.8],fontsize = 12)
ax1.tick_params(axis='x', which='major', pad=30)

ax2 = fig.add_subplot(132, polar=True)
line2,=ax2.plot(angles, stats_48, 'o-', linewidth=2,label='48h',color='brown') # 连线
ax2.fill(angles, stats_48, alpha=0.25,color='brown') # 填充

ax2.set_thetagrids(angles*180/np.pi,labels,fontsize = 18)
ax2.set_rgrids([0.2,0.4,0.6,0.8],fontsize = 12)
ax2.tick_params(axis='x', which='major', pad=30)
ax3 = fig.add_subplot(133, polar=True)
line3,=ax3.plot(angles, stats_72, 'o-', linewidth=2,label='72h',color='green') # 连线
ax3.fill(angles, stats_72, alpha=0.25,color='green') # 填充

ax3.set_thetagrids(angles*180/np.pi,labels,fontsize = 18)
ax3.set_rgrids([0.2,0.4,0.6,0.8],fontsize = 12)
ax3.tick_params(axis='x', which='major', pad=30)
plt.legend(handles=[line1, line2, line3],bbox_to_anchor=(1.00, 1.2), loc=2, borderaxespad=2,frameon=False,fontsize='xx-large')
plt.subplots_adjust(left=0.05, bottom=None, right=None, top=None, wspace=0.5,hspace=0.1)
plt.savefig('./index_5.png')
plt.show()
