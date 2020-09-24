import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from scipy import spatial
import getopt, sys
import os, time
import numpy as np
from netCDF4 import Dataset
import ast 

#
# this is a new version of the diffusion code in 2D
# I decided to rewite from scratch as the plotting and structure were
# rather clunky in the previous code
#

#
# Dq/Dt= C - S - D
# C=Convection
# S=Subsidence
# D=Diffusion
#
# S=constant tau relaxation here
# C=fast relaxation to saturated+detrained IWP
# D=Diffusion at constant K
#
# Convection is an average rate
# 1. distribute this over the day according to diurnal cycle
# 2. Choose where to locate according to
#    a) CRH profile of Bretherton
#    b) A Coldpool inhibition function

# surface nfig out frequency (slows it down a lot!)

def diffusion(fld,a0,a1,ndiff):
    """ diffusion of field using Dufort Frankel explicit scheme"""
    """ argments are fld: 3 slice field"""
    """ a0 and a1 are the DF coefficients"""
    for i in range(ndiff):
        fld[2,:,:]=a0*fld[0,:,:]+a1*(
        np.roll(fld[1,:,:],-1,axis=0) +
        np.roll(fld[1,:,:], 1,axis=0) +
        np.roll(fld[1,:,:],-1,axis=1) +
        np.roll(fld[1,:,:], 1,axis=1))
        fld=np.roll(fld,-1,axis=0) # 2 to 1, 1 to 0 and 0 overwritten
    return(fld)


# PUT default values here in argument list dictionary :-) 
def main(args={"diffK":37500,"tau_sub":20,"crh_ad":16.12,"cin_radius":-99,"diurn_cases":["n"]}):
    """main routine for diff 2d model"""


    global odir
    diffK=args["diffK"]
    tau_sub=args["tau_sub"]
    crh_ad=args["crh_ad"]
    print (args["diurn_cases"],"ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")


    s=args["diurn_cases"]
    print(s[0])
    exit()
    cin_radius=args["cin_radius"] # set to negative num to turn off coldpools

    tab=str(diffK)+"_"+str(tau_sub)[0:6]+"_"+str(crh_ad)+"_"+"_"+str(cin_radius)


    nfig_hr=24
    sfig_day=0


    # cin_radius=20 # radius of coldpools in km (now passed as argument)
    # domain size in m
    
    global domain_x,domain_y, dx, dy
    domain_x=domain_y=500.e3
    dx=dy=2000.
    dxkm=dx/1000.
    cin_radius/=dxkm
    
    # diurnal cycle: "none", "weak", "strong"

    # timestep
    dt=120.
    dtdiff=dt
    ndiff=int(dt/dtdiff)

    # initial column RH
    crh_init=0.8

    # time integration days
    nday=5

    # convective detrained value
    # this is 1+IWP/PW, IWP max ~ 3kg/m**2 for 60kg/m**2
    crh_det=1.05

    # CRH: diffusion K=eps.u.L
    # diffK=0.15*5*50e3    # DEFAULT :  0.4*5*50.e3
    
    # COLDPOOL: diffusion K=eps.u.L
    diffCIN=0.25*10*50.e3 # DEFAULT 0.15*10*20.e3

    # timescale of subsidence and convection
    #tau_sub=20 # days
    tau_cnv=60. # seconds
    tau_cin=3*3600. # 3 hour lifetime for coldpools 
    cnv_lifetime=1800. # e-folding convection detrains for 30mins

    # thresh for cin inhibition 

    # velocity scales
    w_cnv=10.

    # crh_ad from Rushley et al 18, https://doi.org/10.1002/2017GL076296
    #crh_ad=16.12 # trmm v5 
    #crh_ad=14.72 # trmm v7

    # fake diurnal cycle
    diurn_a=0.6
    diurn_p=2
    diurn_o=0.35

    
    odir="../plots/"

    ltest=False
    
    #-------------------
    # derived stuff
    #-------------------
    #
    # sub facs
    #

    tau_sub*=86400. # scale to seconds
    w_sub=15000./tau_sub # subsidence velocity is depth of trop/tau_sub.
    dt_tau_sub=1.0+dt/tau_sub
    dt_tau_cnv=dt/tau_cnv
    dt_tau_cin_fac=1.0+dt/tau_cin
    dt_tau_cin=dt/tau_cin

    try:
        os.mkdir(odir)
    except:
        pass

    # will assume diffusion same in both directions:
    alfa=diffK*dtdiff/(dx*dx)
    alf0=(1.0-4.0*alfa)/(1.0+4.0*alfa) # level zero factor 
    alf1=2.*alfa/(1.0+4.0*alfa) # level 1 factor
    # print(" first alf",diffK,alfa,alf0,alf1)
    alfacin=diffCIN*dtdiff/(dx*dx)
    alfcin0=(1.0-4.0*alfacin)/(1.0+4.0*alfacin) # level zero factor 
    alfcin1=2.*alfacin/(1.0+4.0*alfacin) # level 1 factor

    #print(" cin alf",alfcin0,alfcin1)

    cnv_death=min(dt/cnv_lifetime,1.0)
    nx=int(domain_x/dx)+1 ; ny=int(domain_y/dy)+1
    x1d=np.linspace(0,domain_x,nx)
    y1d=np.linspace(0,domain_x,nx)
    x,y=np.meshgrid(x1d/1000,y1d/1000) # grid in km
    allidx=np.argwhere(np.zeros([nx,ny])<1) # all true

    print ("opening output"),tab

    # open the netcdf files:
    nc1 = Dataset("td_maps_"+tab+".nc", "w", format="NETCDF4")
    nc2 = Dataset("td_stats_"+tab+".nc", "w", format="NETCDF4")

    # dims:                  
    time1=nc1.createDimension("time", None)
    x1=nc1.createDimension("x", ny)
    y1=nc1.createDimension("y", nx)
    nccnt = 0 # counter for ncindex

    # vars
    var_time1 = nc1.createVariable("time","f8",("time",))
    var_x = nc1.createVariable("X","f4",("x",))
    var_y = nc1.createVariable("Y","f4",("y",))
    # two dimensions unlimited
    var_CRH=nc1.createVariable("CRH","f4",("time","y","x",))
    var_CRH.units = "fraction"

    if cin_radius>0:
        var_CIN=nc1.createVariable("CIN","f4",("time","y","x",))
        var_CIN.units = "fraction"

    # timeseries file
    dim_time2 = nc2.createDimension("time", None)
    var_time2 = nc2.createVariable("time","f8",("time",))
    crh_mean = nc2.createVariable("CRH_mean","f8",("time",))
    crh_std = nc2.createVariable("CRH_std","f8",("time",))
    crh_in_new = nc2.createVariable("CRH_new_conv","f8",("time",))
    crh_driest = nc2.createVariable("CRH_driest","f8",("time",))

    nc1.description="2d snapshots"
    nc1.history="Created today "
    nc1.source="Adrian Tompkins (tompkins@ictp.it)"
    nc1.diffK=nc2.diffK=diffK
    nc1.tau_sub=nc2.tau_sub=tau_sub
    nc1.crh_ad=nc2.crh_ad=float(crh_ad)
    nc1.crh_det=nc2.crh_det=crh_det
    nc1.cin_radius=nc2.cin_radius=cin_radius
    nc1.crh_init=nc2.crh_init=crh_init
    nc1.cnv_lifetime=nc2.cnv_lifetime=cnv_lifetime
    nc1.tau_cnv=nc2.tau_cnv=tau_cnv
    nc1.tau_cin=nc2.tau_cin=tau_cin
    nc1.diffCIN=nc2.diffCIN=diffCIN
    nc1.w_cnv=nc2.w_cnv=w_cnv
    
    var_y.units = "km"
    var_x.units = "km"
    var_time1.units=var_time2.units="seconds since 2000-01-01 00:00:00.0"
    var_time1.calendar=var_time2.calendar="gregorian"

    var_y[:]=y1d
    var_x[:]=x1d

    print ("opening file 2")    
    # file 2 is the timeseries file

    # number of timesteps:
    nt=int(nday*86400/dt)
    times=np.arange(0,nt,1)
    days=times*dt/86400.

    # total number of events to distribute
    ncnv_tot=int(nt*nx*ny*w_sub/w_cnv)

    # 3 options of diurnal cycle!
    diurn_opts={}
    diurn_opts["n"]=np.ones(nt)
    diurn_opts["w"]=diurn_a*np.sin(np.pi*2*times*dt/86400.)+1.0
    diurn_opts["s"]=(np.sin(np.pi*2*times*dt/86400.)+1.0)**diurn_p
    diurn_opts["s"]=diurn_opts["s"]/np.mean(diurn_opts["s"]) # mean=1

    #
    # set up plots, timeseries
    #

    for sdiurn in diurn_cases:

        #
        # set up envelope
        #
        pdiurn=diurn_opts[sdiurn]
        pdiurn/=np.sum(pdiurn) # probs must add to 1

        #
        # number of convective events as function of time
        # 
        ncnv=np.bincount(np.random.choice(times,ncnv_tot,p=pdiurn),minlength=nt)
        ncnv_overflow=0 # storage for overflow 
        Nsmth=int(cnv_lifetime/dt) # need to smooth to lifetime of convection
        if Nsmth>1:
            ncnv=uniform_filter1d(ncnv,size=Nsmth)


        # index for convection locations, 0 or 1 
        cnv_idx=np.zeros([nx,ny],dtype=np.int)

        # CIN array for coldpools
        cin=np.zeros([3,nx,ny])

        # crh, 3 time-level DF explicit scheme:
        crh=np.random.normal(loc=crh_init,scale=0.01,size=[3,nx,ny])

        # TEST top hat
        mp=int(nx/2)
        if ltest:
            crh[:,mp-5:mp+5,mp-5:mp+5]=1.0

        dummy_idx=np.arange(0,nx*ny,1)

        ifig=0

        # loop over time
        for it in range(nt):
            if (it*dt)%(24*3600)==0:
                print ("day ",int(it*dt/86400.))
            # explicit diffusion.
            # use np.roll for efficient periodic boundary conditions
            crh=diffusion(crh,alf0,alf1,ndiff)

            #
            # now apply implicit solution for subsidence
            #
            crh[1,:,:]/=dt_tau_sub

            #
            # now apply implicit solution for convection 
            #

            # First calculate residual N to generate this timestep
            ncnv_curr=np.sum(cnv_idx)
            ncnv_new=ncnv[it]+ncnv_overflow-ncnv_curr
            #print("current",ncnv_curr,"ncnv ",ncnv[it]," overflow",ncnv_overflow," new ",ncnv_new)
            ncnv_overflow=0 # overflow accounted for, so reset to zero
            if (ncnv_new<0):
                # we have too many convection events still alive, so we borrow from a future
                # timestep.
                ncnv_overflow=ncnv_new  # store overflow
                ncnv_new=0 #  can't have neg new events 

            #
            # now need to decide where to put the new events, 
            # bretherton updated CRH - with Craig adjustment
            prob_crh=np.exp(crh_ad*crh[1,:,:])-1.0
            
            # fudge to stop 2 conv in one place, coldpool will sort
            prob_crh*=(1-cnv_idx)
            prob_crh/=np.mean(prob_crh)

            # INCLUDE cold pool here:
            #prob_cin=np.where(cin[1,:,:]>cin_thresh,0.0,1.0)
            #prob_cin=1.0-np.power(cin[1,:,:],0.15)
            prob_cin=1.0-cin[1,:,:]
            
            # product of 2:
            prob=prob_crh
            if cin_radius>0:
                prob*=prob_cin # switch off coldpools here:
            prob/=np.sum(prob) # normalized
            prob1d=prob.flatten()

            #
            # sample the index using the prob function
            # and PLACE NEW EVENTS:
            #
            coords=np.random.choice(dummy_idx,ncnv_new,p=prob1d,replace=False)
            new_loc=np.unravel_index(coords,(nx,ny))
            cnv_idx[new_loc]=1 # new events in slice zero
            slice=crh[1,:,:]
            if ncnv_new>0:
                crh_in_new[it]=np.mean(slice[new_loc])
            crh_driest[it]=np.min(slice)


            # cnv_idx[mp,mp]=1 # TEST

            # update humidity
            # collape conv array again # Q_Detrain where conv, zero otherwise
            crh[1,:,:]=(crh[1,:,:]+cnv_idx*crh_det*dt_tau_cnv)/(1.0+cnv_idx*dt_tau_cnv)

            #
            # update coldpool here
            #
            if cin_radius>0:
                cnv_coords=np.argwhere(cnv_idx)
                ncnv_curr=np.sum(cnv_idx)
                if ncnv_curr>0:
                    #distlist=[]
                    #for ioff in [-nx,0,nx]:
                    #    for joff in [-ny,0,ny]:
                    #        j=cnv_coords.copy()
                    #        j[:,0]+=ioff
                    #        j[:,1]+=joff
                    #        distlist.append(np.amin(scidist.cdist(allidx,j,metric='euclidean'),1))
                    #cnvdst=np.amin(np.stack(distlist),0).reshape(nx,ny)
                    #cnvdst*=dxkm
                    for xoff in [0,nx,-nx]:
                        for yoff in [0,-ny,ny]:
                            if xoff==0 and yoff==0:
                                j9=cnv_coords.copy()
                            else:
                                jo=cnv_coords.copy()
                                jo[:,0]+=xoff
                                jo[:,1]+=yoff
                                j9=np.vstack((j9,jo))
                    tree=spatial.cKDTree(j9)
                    cnvdst,minidx=tree.query(allidx)
                    cnvdst=cnvdst.reshape([nx,ny])
                    
                else:
                    cnvdst=np.ones([nx,ny])*1.e6

                maskcin=np.where(cnvdst<cin_radius,1,0)
                
                cin[1,:,:]=cin[1,:,:]+maskcin # all conv points sets to 1
                cin=np.clip(cin,0,1)

            # cin[1,:,:]*=dt_tau_cin_fac # implicit
                cin[1,:,:]-=dt_tau_cin # explicit 
                cin=diffusion(cin,alfcin0,alfcin1,ndiff)            
                cin=np.clip(cin,0,1)

            #
            # random death of cells.
            # 
            mask=np.where(np.random.uniform(size=(nx,ny))<=cnv_death,0,1)
            cnv_idx*=mask

            #
            cnv_loc=np.argwhere(cnv_idx==1)    

            # netcdf output for timeseries:
            var_time2[it]=it*dt
            crh_mean[it]=np.mean(crh[1,:,:])
            crh_std[it]=np.std(crh[1,:,:])

            day=it*dt/86400
            if (it*dt)%(nfig_hr*3600)==0:
                var_time1[nccnt]=it*dt
                var_CRH[nccnt,:,:]=crh[1,:,:]     
                if cin_radius>0:
                    var_CIN[nccnt,:,:]=cin[1,:,:]     


                nccnt+=1
              
                
    nc1.close()
    nc2.close()

if __name__ == "__main__":

    # default values: crhad from bretherton et al update 2017
    diffK=37500. # m2/s
    crh_ad=16.12
    tau_sub=20. # days!
    cin_radius=-99. # switched off by default
    diurn_cases="n"
    
    arglist=["help","diffK=","crh_ad=","tau_sub=","odir=","cin_radius=","nfig_hr="]
    try:
        opts, args = getopt.getopt(sys.argv[1:],"h",arglist)
    except getopt.GetoptError:
        print ("args are:",arglist)  
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print ("check out this list: ",arglist)
            sys.exit()
        elif opt in ("--diffK"):
            diffK = float(arg)
        elif opt in ("--crh_ad"):
            crh_ad = float(arg)
        elif opt in ("--cin_radius"):
            cin_radius = float(arg)
        elif opt in ("--tau_sub"):
            tau_sub = float(arg)
        elif opt in ("--nfig_hr"):
            nfig_hr = int(arg)
        elif opt in ("--odir"):
            odir = arg

    print ("diurn_cases",diurn_cases)

    # pass args as a dictionary to ensure one arg only, two opts are missing, add later
    args={"diffK":diffK,"tau_sub":tau_sub,"crh_ad":crh_ad,"cin_radius":cin_radius,
          "diurn_cases":diurn_cases}    

    main(args)
