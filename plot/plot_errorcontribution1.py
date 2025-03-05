import pandas as pd
#计算每个台风的误差贡献率
import math
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import chain
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

def compute_dis(m,n,lat_fore,lon_fore,lat_true,lon_true):
	dis_sum=0
	dis_ds=[]
	for i in range(m,n):
		fore_lat=lat_fore[i]
		fore_lon=lon_fore[i]
		true_lat=lat_true[i]
		true_lon=lon_true[i]
		dis=getDistance(fore_lat, fore_lon, true_lat, true_lon)
		dis_sum+=dis
		dis_ds.append(dis)
	return dis_sum,dis_sum/(n-m),dis_ds

def compute_latlon(m,n,lat_fore,lon_fore,lat_true,lon_true):
	dis_lat=[]
	dis_lon=[]
	for i in range(m,n):
		fore_lat=lat_fore[i]
		fore_lon=lon_fore[i]
		true_lat=lat_true[i]
		true_lon=lon_true[i]
		dt_lat=fore_lat-true_lat
		dt_lon = fore_lon - true_lon
		dis_lat.append(dt_lat)
		dis_lon.append(dt_lon)
	return np.stack(dis_lat),np.stack(dis_lon)

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

def year_split(year,hours):
	i_index_name=[]
	j_index_all = []
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
		j_index = []
		if str(dataframe['date'][m])[0:4] == str(year) and n - hh > m + 4:
			i_index_name.append(dataframe['name'][m])
			for j in range(m + 4, n - hh):
				j_index.append(j)
			j_index_all.append(j_index)
	return j_index_all,i_index_name

j_index_all_12=[]
j_index_all_24=[]
j_index_all_36=[]
j_index_all_48=[]
j_index_all_60=[]
j_index_all_72=[]
i_index_name_24=[]
i_index_name_48=[]
i_index_name_72=[]
for year in range(2022,2025):
	j_index_all_12.append(year_split(year, 12)[0])
	j_index_all_24.append(year_split(year,24)[0])
	j_index_all_36.append(year_split(year,36)[0])
	j_index_all_48.append(year_split(year,48)[0])
	j_index_all_60.append(year_split(year,60)[0])
	j_index_all_72.append(year_split(year,72)[0])
	i_index_name_24.append(year_split(year,24)[1])
	i_index_name_48.append(year_split(year,48)[1])
	i_index_name_72.append(year_split(year,72)[1])



def compute_contribute(hours,j_index_all):
	str_h=str(hours)
	lat_fore=fore_track[str_h]['lat']
	lon_fore=fore_track[str_h]['lon']

	lat_true=true_track[str_h]['lat']
	lon_true=true_track[str_h]['lon']
	m=0
	dis_mean_all=[]
	dis_lat_all=[]
	dis_lon_all=[]
	dis_sum_all=[]
	for j_index_all_y in j_index_all:
		dis_mean_ds=[]
		dis_lat_ds = []
		dis_lon_ds=[]
		dis_sum_ds=[]
		for kk in range(len(j_index_all_y)):
			n=m+len(j_index_all_y[kk])
			dis_sum,dis_mean,dis_ds=compute_dis(m, n, lat_fore, lon_fore, lat_true, lon_true)
			dis_lat,dis_lon=compute_latlon(m, n, lat_fore, lon_fore, lat_true, lon_true)
			m=n
			dis_sum_ds.append(dis_ds)
			dis_mean_ds.append(dis_mean)
			dis_lat_ds.append(dis_lat)
			dis_lon_ds.append(dis_lon)
		dis_mean_all.append(dis_mean_ds)
		dis_lat_all.append(dis_lat_ds)
		dis_lon_all.append(dis_lon_ds)
		dis_sum_all.append(list(chain.from_iterable(dis_sum_ds)))
	return dis_mean_all,dis_lat_all,dis_lon_all,dis_sum_all
dis_mean_all_12,dis_lat_all_12,dis_lon_all_12,dis_sum_all_12=compute_contribute(12,j_index_all_12)
dis_mean_all_24,dis_lat_all_24,dis_lon_all_24,dis_sum_all_24=compute_contribute(24,j_index_all_24)
dis_mean_all_36,dis_lat_all_36,dis_lon_all_36,dis_sum_all_36=compute_contribute(36,j_index_all_36)
dis_mean_all_48,dis_lat_all_48,dis_lon_all_48,dis_sum_all_48=compute_contribute(48,j_index_all_48)
dis_mean_all_60,dis_lat_all_60,dis_lon_all_60,dis_sum_all_60=compute_contribute(60,j_index_all_60)
dis_mean_all_72,dis_lat_all_72,dis_lon_all_72,dis_sum_all_72=compute_contribute(72,j_index_all_72)

typhoon_name_ds=[]
typhoon_name_ds.append(['MALAKAS(01)', 'MEGI(02)', 'CHABA(03)', 'AERE(04)', 'SONGDA(05)', 'TRASES(06)','MEARI(08)','MA-ON(09)', 'TOKAGE(10)', 'HINNAMNOR(11)', 'MUIFA(12)', 'MERBOK(13)', 'NANMADOL(14)', 'TALAS(15)', 'NORU(16)', 'KULAP(17)', 'ROKE(18)','SONCA(19)','NESAT(20)','HAITANG(21)','NALGAE(22)','BANYAN(23)','YAMANEKO(24)','PAKHAR(25)'])
typhoon_name_ds.append(['SANVU(01)','MAWAR (02)','GUCHOL (03)','TALIM (04)','DOKSURI (05)','KHANUN (06)','LAN (07)','DORA (08)',
			  'SAOLA (09)','DAMREY (10)','HAIKUI (11)','KIROGI (12)','YUN-YEUNG(13)','KOINU (14)','BOLAVEN (15)','SANBA(16)', 'JELAWAT(17)'])
typhoon_name_ds.append(['EWINIAR(01)', 'GAEMI(03)', 'PRAPIROON(04)','MARIA(05)', 'AMPIL(07)', 'WUKONG(08)','SHANSHAN(10)', 'YAGI(11)','LEEPI(12)', 'BEBINCA(13)', 'PULASAN(14)','CIMARON(16)', 'JEBI(17)', 'KRATHON(18)', 'BARIJAT(19)', 'TRAMI(20)', 'KONG-REY(21)', 'YINXING(22)', 'TORAJI(23)', 'MAN-YI(24)', 'USAGI(25)','PABUK(26)'])

# yy=0
# pos1=6
# pos2=7
# year=2022
typhoon_name_2022=typhoon_name_ds[0]
typhoon_name_2023=typhoon_name_ds[1]
typhoon_name_2024=typhoon_name_ds[2]

dis_mean_24_ds_2022=dis_mean_all_24[0]
dis_mean_24_ds_2023=dis_mean_all_24[1]
dis_mean_24_ds_2024=dis_mean_all_24[2]
dis_ds_lat_2022=dis_lat_all_24[0]
dis_ds_lat_2023=dis_lat_all_24[1]
dis_ds_lat_2024=dis_lat_all_24[2]
dis_ds_lon_2022=dis_lon_all_24[0]
dis_ds_lon_2023=dis_lon_all_24[1]
dis_ds_lon_2024=dis_lon_all_24[2]
i_ds_name_2022=i_index_name_24[0]
i_ds_name_2023=i_index_name_24[1]
i_ds_name_2024=i_index_name_24[2]

def swapPositions(list, pos1, pos2):
	list[pos1], list[pos2] = list[pos2], list[pos1]
	return np.array(list)

for i in i_ds_name_2022:
	if i=='UNNAMED':
		i_ds_name_2022.remove(i)

for i in i_ds_name_2023:
	if i=='UNNAMED':
		i_ds_name_2023.remove(i)

for i in i_ds_name_2024:
	if i=='UNNAMED':
		i_ds_name_2024.remove(i)


name_2022_index=[]
name_2023_index=[]
name_2024_index=[]

for i in i_ds_name_2022:
	name_2022_index.append(i_ds_name_2022.index(i))
for i in i_ds_name_2023:
	name_2023_index.append(i_ds_name_2023.index(i))
for i in i_ds_name_2024:
	name_2024_index.append(i_ds_name_2024.index(i))

name_2022_index=list(np.arange(0,25,1))
name_2022_index.remove(20)
temp17 = name_2022_index[17]
temp18 = name_2022_index[18]
temp19 = name_2022_index[19]
name_2022_index[19] = temp17
name_2022_index[18] = temp19
name_2022_index[17] = temp18
print(np.array(i_index_name_24[2])[name_2024_index])
print(typhoon_name_2024)

dis_mean_2022_ds_new=np.array(dis_mean_24_ds_2022)[name_2022_index]
dis_mean_2023_ds_new=swapPositions(np.array(dis_mean_24_ds_2023)[name_2023_index], 6, 7)
dis_mean_2024_ds_new=np.array(dis_mean_24_ds_2024)[name_2024_index]
dis_ds_lat_2022_new=np.array(dis_ds_lat_2022)[name_2022_index]
dis_ds_lat_2023_new=swapPositions(np.array(dis_ds_lat_2023)[name_2023_index],6,7)
dis_ds_lat_2024_new=np.array(dis_ds_lat_2024)[name_2024_index]
dis_ds_lon_2022_new=np.array(dis_ds_lon_2022)[name_2022_index]
dis_ds_lon_2023_new=swapPositions(np.array(dis_ds_lon_2023)[name_2023_index], 6, 7)
dis_ds_lon_2024_new=np.array(dis_ds_lon_2024)[name_2024_index]


dis_sum_2022_ds=dis_sum_all_24[0]
dis_sum_2023_ds=dis_sum_all_24[1]
dis_sum_2024_ds=dis_sum_all_24[2]


typhoon_num_2022=[i.split('(')[1][0:2] for i in typhoon_name_2022]
typhoon_num_2023=[i.split('(')[1][0:2] for i in typhoon_name_2023]
typhoon_num_2024=[i.split('(')[1][0:2] for i in typhoon_name_2024]

def plot_contribution(i,dis_mean_24_ds_new,dis_sum_24_ds,typhoon_name,typhoon_num,colors):

	ax = plt.subplot(3, 1, i)
	mean_all = sum(dis_sum_24_ds) / len(dis_sum_24_ds)

	data = list(np.array(dis_mean_24_ds_new) / mean_all)

	data.reverse()
	ax.barh(range(len(data)), data, tick_label=typhoon_name, color=colors)
	ax.set_xlabel('Error Contribution Rate', fontsize=15, labelpad=8)
	plt.axvline(1,ls='--',color='r')
	ax.set_xlim([0,2.3])
	ax.set_xticks([0,0.5,1,1.5,2])
	# ax.set_xticklabels(typhoon_num)
def plot_box(i, dis_ds_lat_24_new,typhoon_name,typhoon_num,list_range):
	x = dis_ds_lat_24_new
	p = np.arange(len(typhoon_name))
	position = tuple(p)
	ax = plt.subplot(3, 2, i)
	y = ax.boxplot(x, showmeans=True, showfliers=True, widths=0.5, positions=position,
				     flierprops=dict(marker='o', markerfacecolor="none", markersize=6,markeredgecolor='black'),
					 meanprops={"marker": "^", "markeredgecolor":'blue',"markerfacecolor": "none", "markersize": 8},
					 boxprops={'color': '#015699'}, medianprops={'color': 'red'},
					 whiskerprops={'linestyle': '--', 'linewidth': 0.7})  # 绘制箱形图，设置异常点大小、样式等
	plt.ylabel('Error (°)', fontsize=15, labelpad=8)
	plt.xlabel('Typhoon Numer', fontsize=15, labelpad=8)
	ax.set_xticks(np.arange(0, len(typhoon_name), 1))
	ax.set_xticklabels(typhoon_num)
	ax.set_yticks(list_range)
	e = Line2D([0], [0], color='red', linewidth=1, linestyle='-', label='Median')
	f = Line2D([], [], color='blue', marker='^',markerfacecolor="none", markersize=10, label='Mean', ls='')
	plt.legend(handles=[e, f],shadow=True,fontsize='small')



# print(dis_sum_24_ds_new,dis_sum_48_ds_new,dis_sum_72_ds_new)
fig = plt.figure(figsize=(8, 16), dpi=300)
plt.rcParams.update({'font.family': 'Times New Roman','font.size': 15})
# matplotlib.rc('font', family='SimHei', weight='bold')
typhoon_name_2022.reverse()
typhoon_name_2023.reverse()
typhoon_name_2024.reverse()
plot_contribution(1,dis_mean_2022_ds_new,dis_sum_2022_ds,typhoon_name_2022,typhoon_num_2022,'blue')
plot_contribution(2,dis_mean_2023_ds_new,dis_sum_2023_ds,typhoon_name_2023,typhoon_num_2023,'orange')
plot_contribution(3,dis_mean_2024_ds_new,dis_sum_2024_ds,typhoon_name_2024,typhoon_num_2024,'grey')

plt.subplots_adjust(left=0.25, bottom=0.1, right=0.9, top=0.98, wspace=0.1,hspace=0.2)
plt.savefig(f'./typhoons_contribution.png')
plt.show()


# fig = plt.figure(figsize=(15, 14), dpi=300)
# plt.rcParams.update({'font.family': 'Times New Roman','font.size': 15})
# plot_box(1, dis_ds_lat_2022_new,typhoon_name_2022,typhoon_num_2022,np.arange(-5,5.5,2))
# plot_box(3, dis_ds_lat_2023_new,typhoon_name_2023,typhoon_num_2023,np.arange(-5,5.5,2))
# plot_box(5, dis_ds_lat_2024_new,typhoon_name_2024,typhoon_num_2024,np.arange(-5,5.5,2))
# plot_box(2, dis_ds_lon_2022_new,typhoon_name_2022,typhoon_num_2022,np.arange(-5,5.5,2))
# plot_box(4, dis_ds_lon_2023_new,typhoon_name_2023,typhoon_num_2023,np.arange(-5,5.5,2))
# plot_box(6, dis_ds_lon_2024_new,typhoon_name_2024,typhoon_num_2024,np.arange(-5,5.5,2))
# plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.98, wspace=0.2,hspace=0.25)
# plt.savefig(f'./box.png')
# plt.show()