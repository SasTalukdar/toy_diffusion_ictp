#
# plot code janked from main code
#
import numpy as np 
import matplotlib.pyplot as plt
from netCDF4 import Dataset

odir="./"
csfont = {'fontname':'Helvetica'}

def map_plot(ax,fld,title,day,exp,vmin,vmax,loc,d2c1):
    print (title)
    img=ax.pcolormesh(x,y,fld,cmap='Spectral',vmin=vmin,vmax=vmax)
    ax.set_title("day "+str(day),**csfont)
    #ax.axis([x.min(), x.max(), y.min(), y.max()])
    if axidx[0]==1:
        ax.set_xlabel('km', fontsize=15)
    if axidx[1]==0:
        ax.set_ylabel('km', fontsize=15)
    #ax.set_xlim(0,int(domain_x/1000.))
    #ax.set_ylim(0,int(domain_y/1000.))
    print(d2c1)
    cs=ax.contour(x,y,d2c1,[2],colors='white')
    ax.set_xticks(np.arange(0,600,100))
    ax.set_yticks(np.arange(0,600,100))
    if False:
        ax.scatter([y*2 for x,y in convidx],[x*2 for x,y in convidx],s=5,marker="s",zorder=1,color="black",edgecolors="white")
    plt.colorbar(img,ax=ax)
    plt.subplots_adjust(left=0.12,right=0.82, bottom=0.1,top=0.9,wspace=0.25,hspace=0.4)


    # other plot stuff

slices=[i*8 for i in [5,20,30,100]]

dir="/home/netapp-clima/users/tompkins/diffusion/"
file="td_maps_diffK7000_tausub15_crhad16.12_cin_radius6_diurn0.nc"

ds=Dataset(dir+file)

crh=ds.variables["CRH"]
d2c=ds.variables["D2C"]
# division needed due to bug 
x=np.array(ds.variables["X"])/2000
y=np.array(ds.variables["Y"])/2000 
time=np.array(ds.variables["time"])

cnv_loc=0

sday=10
sdiurn=2
ncols=nrows=2
fig,ax=plt.subplots(nrows=nrows,ncols=ncols)
print (slices)
print (time.shape)


for idx,sday in enumerate(slices):
    axidx=np.unravel_index(idx,(nrows,ncols))
    print (axidx[0])
    map_plot(ax[axidx],crh[sday,:,:],"CRH",time[sday]/86400.,sdiurn,0.3,1.05,cnv_loc,d2c[sday,:,:])
plt.savefig(odir+file[:-3]+".png")
plt.close(fig)


#map_plot(x,y,cin[1,:,:],"CIN",sday,ifig,sdiurn,0,1,cnv_loc)
#map_plot(x,y,prob,"P",sday,ifig,sdiurn,0,np.max(prob),cnv_loc)
#map_plot(x,y,prob_cin,"P-CIN",sday,ifig,sdiurn,0,np.max(prob_cin),cnv_loc)
#map_plot(x,y,prob_crh,"P-CRH",sday,ifig,sdiurn,0,np.max(prob_crh),cnv_loc)

# correct these surface plots
if False:
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot_surface(x,y,crh[1,:,:],cmap='viridis', edgecolor='none')
    ax.set_title('CRH: day '+str(ifig*nfig_hr/24))
    ax.set_zlim([0.4,1])
    fig.savefig("crh_3d_"+str(ifig).zfill(3)+".png")
    plt.close(fig)

    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot_surface(x,y,cin[1,:,:],cmap='viridis', edgecolor='none')
    ax.set_title('CIN p: day '+str(ifig*nfig_hr/24))
    ax.set_zlim([0,1])
    fig.savefig("cin_3d_"+str(ifig).zfill(3)+".png")
    plt.close(fig)

