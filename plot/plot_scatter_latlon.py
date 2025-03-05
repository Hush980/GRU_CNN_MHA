import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
import json
with open('../results/results_drop/fore_json.json','r',encoding='utf8')as fp:
	fore_track = json.load(fp)

with open('../results/results_drop/truth_json.json','r',encoding='utf8')as fp:
    truth_track = json.load(fp)



def compute_gaussian(x,y):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x=np.array(x)
    y = np.array(y)
    x_new, y_new, z_new = x[idx], y[idx], z[idx]
    return x_new,y_new,z_new

# index=[0,0,1,1,3,3,7,7,11,11]
# number=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(i)','(j)','(i)','(j)']
# fig = plt.figure(figsize=(14, 22), dpi=300)
# plt.rcParams.update({'font.family': 'Times New Roman','font.size': 18})
# # matplotlib.rc("font",family='SimHei')
# for m in range(10):
#     ax=plt.subplot(5,2,m+1)
#     i=index[m]
#     hour=(i+1)*6
#     if m%2==1:
#         k=80
#         n=200
#         x=np.arange(k,n)
#         y=x
#         plt.plot(x,y,color='black')
#         lon_true_new, lon_pred_new, lon_density = compute_gaussian(lon_true[:,i], lon_pred[:,i])
#         im=plt.scatter(lon_true_new,lon_pred_new,c=lon_density,marker='.',cmap='jet',vmin=min(lon_density),vmax=max(lon_density))
#         scc='%.6f' %(np.corrcoef(lon_true_new, lon_pred_new)[0,1])
#         ssd='%.3f' %(np.std(abs(lon_true_new-lon_pred_new)))
#         rmse='%.3f' %(mean_squared_error(lon_true_new,lon_pred_new, squared=False))
#         title_name='经度'
#         plt.xticks(np.arange(80,220,40))
#         plt.yticks(np.arange(80,220,40))
#         hour=(index[m-1]+1)*6
#         plt.text(76,195,number[m])
#         plt.text(79,184,f'R:   {scc}')
#         plt.text(79, 173, f'STD:   {ssd}')
#         plt.text(79, 162, f'RMSE: {rmse}')
#
#         plt.ylabel('Predicted longitude(°N)',fontsize=20,labelpad=13,fontfamily='Times New Roman')
#         if m==9:
#             plt.xlabel('True longitude(°N)',fontsize=20,labelpad=10,fontfamily='Times New Roman')
#             ax.set_xticklabels(['80', '120', '160', '200'])
#         else:
#             ax.set_xticklabels([' ', ' ', ' ', ' '])
#     else:
#         k=0
#         n=70
#         x=np.arange(k,n)
#         y=x
#         plt.plot(x,y,color='black')
#         lat_true_new, lat_pred_new, lat_density = compute_gaussian(lat_true[:, i], lat_pred[:, i])
#         im=plt.scatter(lat_true_new,lat_pred_new,c=lat_density,marker='.',cmap='jet',vmin=min(lat_density),vmax=max(lat_density))
#         scc='%.6f' %(np.corrcoef(lat_true_new, lat_pred_new)[0,1])
#         ssd='%.3f' %(np.std(abs(lat_true_new-lat_pred_new)))
#         rmse='%.3f' %(mean_squared_error(lat_true_new,lat_pred_new, squared=False))
#         plt.xticks(np.arange(0,70,20))
#         plt.yticks(np.arange(0,70,20))
#         title_name='纬度'
#         hour=(index[m]+1)*6
#         plt.text(-2,67,number[m])
#         plt.text(1,60,f'R:   {scc}')
#         plt.text(1, 53, f'STD:   {ssd}')
#         plt.text(1, 46, f'RMSE: {rmse}')
#         plt.ylabel('{a}h\n\nPredicted latitude(°E)'.format(a=hour),fontsize=20,labelpad=13,fontfamily='Times New Roman')
#         if m==8:
#             plt.xlabel('True latitude(°E)',fontsize=20,labelpad=10,fontfamily='Times New Roman')
#             # ax.set_xticklabels(['80','100','120','140','160','180','200'])
#             ax.set_xticklabels(['0','20','40','60'])
#         else:
#             ax.set_xticklabels([' ',' ',' ',' ',' '])
# position = fig.add_axes([0.1, 0.91, 0.8, 0.02])
# fc=fig.colorbar(im,cax=position,shrink=1,extend='both',orientation='horizontal',ticks=np.linspace(min(lat_density), max(lat_density),0),label='Sparse                                                                                                                                              Dense')
# # ax_a=fc.ax
# # ax_a.set_title('ws (m/s)',pad=10,fontsize=20)
# # ax.set_title('{a}  第{b:g}h的预测{c}与实际{c}的散点分布图'.format(a=number[m],b=hour,c=title_name),x=0.5,y=1.1)
# plt.subplots_adjust(left=None, bottom=None, right=0.89, top=None, wspace=0.25,hspace=0.1)
# plt.savefig('./plot/track_scatter.png')
# plt.show()

index=np.arange(12)
number=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
fig = plt.figure(figsize=(20, 12), dpi=300)
plt.rcParams.update({'font.family': 'Times New Roman','font.size': 18})
for m in range(12):
    ax=plt.subplot(3,4,m+1)
    i=index[m]
    hour=(i+1)*6
    k=0
    n=70
    x=np.arange(k,n)
    y=x
    plt.plot(x,y,color='black')
    lat_true = truth_track[str(hour)]['lat']
    lat_pred = fore_track[str(hour)]['lat']
    lat_true_new, lat_pred_new, lat_density = compute_gaussian(lat_true[:], lat_pred[:])
    im=plt.scatter(lat_true_new,lat_pred_new,c=lat_density,marker='.',cmap='jet',vmin=min(lat_density),vmax=max(lat_density))
    scc='%.6f' %(np.corrcoef(lat_true_new, lat_pred_new)[0,1])
    ssd='%.3f' %(np.std(abs(lat_true_new-lat_pred_new)))
    rmse='%.3f' %(mean_squared_error(lat_true_new,lat_pred_new, squared=False))
    plt.xticks(np.arange(0,70,20))
    plt.yticks(np.arange(0,70,20))
    title_name='纬度'
    hour=(index[m]+1)*6
    plt.text(-2,67,number[m])
    plt.text(1,60,f'R:   {scc}')
    plt.text(1, 53, f'STD:   {ssd}')
    plt.text(1, 46, f'RMSE: {rmse}')
    if m in [0,4,8]:
        ax.set_yticklabels(['0','20','40','60'])
        plt.ylabel('Predicted latitude(°E)',fontsize=20,labelpad=13,fontfamily='Times New Roman')
    else:
        ax.set_yticklabels([' ', ' ', ' ', ' '])
    if m in [8,9,10,11]:
        plt.xlabel('True latitude(°E)',fontsize=20,labelpad=10,fontfamily='Times New Roman')
        ax.set_xticklabels(['0','20','40','60'])
    else:
        ax.set_xticklabels([' ', ' ', ' ', ' '])
position = fig.add_axes([0.2, 0.91, 0.6, 0.02])
fc=fig.colorbar(im,cax=position,shrink=1,extend='both',orientation='horizontal',ticks=np.linspace(min(lat_density), max(lat_density),0),label='Sparse                                                                                                                                                   Dense')
plt.subplots_adjust(left=0.08, bottom=None, right=0.95, top=None, wspace=0.2,hspace=0.1)
plt.savefig('./lat_scatter.png')
plt.show()

index=np.arange(12)
number=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
fig = plt.figure(figsize=(20, 12), dpi=300)
plt.rcParams.update({'font.family': 'Times New Roman','font.size': 18})
for m in range(12):
    ax=plt.subplot(3,4,m+1)
    i=index[m]
    hour=(i+1)*6
    k=80
    n=200
    x=np.arange(k,n)
    y=x
    plt.plot(x,y,color='black')

    lon_true = truth_track[str(hour)]['lon']
    lon_pred = fore_track[str(hour)]['lon']
    lon_true_new, lon_pred_new, lon_density = compute_gaussian(lon_true[:], lon_pred[:])
    im=plt.scatter(lon_true_new,lon_pred_new,c=lon_density,marker='.',cmap='jet',vmin=min(lon_density),vmax=max(lon_density))
    scc='%.6f' %(np.corrcoef(lon_true_new, lon_pred_new)[0,1])
    ssd='%.3f' %(np.std(abs(lon_true_new-lon_pred_new)))
    rmse='%.3f' %(mean_squared_error(lon_true_new,lon_pred_new, squared=False))
    title_name='经度'
    plt.xticks(np.arange(80,260,40))
    plt.yticks(np.arange(80,260,40))
    if m in [0,1,2,3]:
        plt.text(76,239,number[m])
        plt.text(79,224,f'R:   {scc}')
        plt.text(79, 209, f'STD:   {ssd}')
        plt.text(79, 194, f'RMSE: {rmse}')
    if m in [4,5,6,7]:
        plt.text(76,235,number[m])
        plt.text(79,220,f'R:   {scc}')
        plt.text(79, 205, f'STD:   {ssd}')
        plt.text(79, 190, f'RMSE: {rmse}')
    if m in [8,9,10,11]:
        plt.text(76,233,number[m])
        plt.text(79,216,f'R:   {scc}')
        plt.text(79, 200, f'STD:   {ssd}')
        plt.text(79, 184, f'RMSE: {rmse}')
    if m in [0,4,8]:
        ax.set_yticklabels(['80', '120', '160', '200','240'])
        plt.ylabel('Predicted longitude(°N)',fontsize=20,labelpad=13,fontfamily='Times New Roman')
    else:
        ax.set_yticklabels([' ', ' ', ' ', ' ',' '])
    if m in [8,9,10,11]:
        plt.xlabel('True longitude(°N)',fontsize=20,labelpad=10,fontfamily='Times New Roman')
        ax.set_xticklabels(['80', '120', '160', '200','240'])
    else:
        ax.set_xticklabels([' ', ' ', ' ', ' ',' '])
position = fig.add_axes([0.2, 0.91, 0.6, 0.02])
fc=fig.colorbar(im,cax=position,shrink=1,extend='both',orientation='horizontal',ticks=np.linspace(min(lon_density), max(lon_density),0),label='Sparse                                                                                                                                                   Dense')
plt.subplots_adjust(left=0.08, bottom=None, right=0.95, top=None, wspace=0.2,hspace=0.1)
plt.savefig('./lon_scatter.png')
plt.show()