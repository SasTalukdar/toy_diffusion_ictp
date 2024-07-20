import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from scipy import spatial
from scipy.linalg import solve_circulant
import sys
from datetime import datetime
import os, time
import numpy as np
from netCDF4 import Dataset
import ast 
import getargs
import subprocess
from tqdm import tqdm
from scipy.ndimage import convolve

#A function to create a mask for the CAPE and CIN radius
def create_time_dependent_mask(cnv_loc, memory, cin_radius, cp_vel, cp_init_delay, nx, ny, dxkm, dt):
    '''
    Input parameters:
        cnv_loc: currently active convection locations, 
        memory:  lifetime duration of each convective event, 
        cin_radius: radius of downdraft, 
        cp_vel: downdraft velocity, 
        cp_init_delay: delay for the initiation of convection, 
        nx, ny: domain size, 
        dxkm: spatial resolution, 
        dt: temporal resolution
    
    Output parameters:
        cinmask: mask for downdraft
        capemask: mask for the gustfront
    '''
    
    capemask=np.zeros([nx,ny])
    cinmask=np.zeros([nx,ny])
    cin_radius_m=cin_radius*1000 #Convert to meters
    # Includes all the grid points (will be used in the computation of the distances from the updraft location, = 0 in case the cell itself is developing convection)
    allidx=np.argwhere(np.zeros([nx,ny])<1)
    # As each convective cell has a different lifetime and hence different radius of influence, the mask is updated based on one updraft at a time
    for loc in cnv_loc:
        #The procedure takes into account the doubly-periodic nature of the computational domain
        for xoff in [0,nx,-nx]:
            for yoff in [0,-ny,ny]:
                if xoff==0 and yoff==0:
                    j9=loc.copy()
                else:
                    jo=loc.copy()
                    jo[0]+=xoff
                    jo[1]+=yoff
                    j9=np.vstack((j9,jo))
        tree=spatial.cKDTree(j9)
        cnvdst,minidx=tree.query(allidx)
        cnvdst=cnvdst.reshape([nx,ny])
        cnvdst*=dxkm*1000
        # The delay in activation of CIN and CAPE is considered
        if str(loc) not in memory:
            lt=0
        else:
            lt=memory[str(loc)]*dt #lifetime, seconds
        lt=lt-cp_init_delay
        if lt<0:
            lt=0
        if lt>0:
            capemask[cnvdst<=(3*(cin_radius_m*cin_radius_m*cp_vel*lt)**(1/3)+cin_radius_m)]=1
            cinmask[cnvdst<=cin_radius_m]=1
    return cinmask,capemask

# A function to detect the edge of the CAPE mask and update it with 1's in the edge and 0 everywhere else
def find_0_nearby_1(x):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors_are_1 = convolve((x == 1).astype(int), kernel, mode='constant', cval=0) > 0
    x[(x == 0) & neighbors_are_1] = 2
    x[x==1]=0
    x[x==2]=1
    return x

#The model's governing equation is dR/dt= C - D - S, where R is column total water relative humidity (CRH), C is the convective moistening term, D represents the spatial moisture exchange and S is the subsidence drying. C is modeled as a fast relaxation process to detrained water vapour and cloud condensate, D according to a down-gradient approach with constant coefficient K, S as a relaxation term with uniform timescale towards a dry atmosphere. C does not act on the entire domain: the convection occurs only in the grid cells where an indicator function is activated, based on a random sampling from a non-uniform, CRH-dependent probability distribution function (PDF). The PDF is derived from the nonlinear moisture-precipitation relationship by Bretherton et al. (2004), revised by Rushley et al. (2018). It is also possible to distribute convective events over the day according to a diurnal cycle representation and account for the inhibition effect of cold pools.    

### KEY RUN DEFAULT VALUES ###
def defaults():
    pars={}
    
    #Horizontal moisture diffusion coefficient (same in both directions, in m**2/s)
    pars["diffK"]=20000.
    
    #Steepness of the exponential function governing the choice of convective locations. Default is from TRMM retrieval version 7 (see Rushley et al. (2018), https://doi.org/10.1002/2017GL076296; see also Bretherton et al. (2004), https://doi.org/10.1175/1520-0442(2004)017<1517:RBWVPA>2.0.CO;2, and references therein)
    pars["crh_ad"]=14.72
    
    #Subsidence drying timescale (in days, not seconds!)
    pars["tau_sub"]=16.
    
    #Convective inhibition radius (km) due to the effect of cold pols. Setting cin_radius to a negative value switches off cold pools by default.
    pars["cin_radius"]=1

    #Option to include the diurnal cycle, if = 0 no diurnal cycle is considered
    pars["diurn_opt"]=0

    #Updrafts: vertical velocity, timescale of convective moistening (in seconds)
    pars["w_cnv"]=10.
    pars["tau_cnv"]=60.

    #Convective detrained value (saturation + detrained condensate, references for default value are from CloudSat)
    pars["crh_det"]=1.05
    
    #Average lifetime of a convective event
    pars["cnv_lifetime"]=1800.
    
    #Coldpools: diffusion coefficient (m**2/s) and lifetime (seconds)
    pars["diffCIN"]=10000 #0.25*10*50.e3
    pars["tau_cin"]=7200 #3*3600.
    
    #Sas's doing
    #Coldpools: diffusion coefficient (m**2/s) and lifetime (seconds)
    pars["diffCAPE"]=12000 #0.25*10*50.e3
    pars["tau_cape"]=600.
    pars["downdraft_velocity"]=3 #m/s
    pars["cp_init_delay"]=10*60 #seconds
    pars["vel_coeff"]=1

    #Wind shear
    pars["shear"]=8 #km
    
    #Different diurnal cycle specifications (active only if diurn_opt != 0)  
    pars["diurn_a"]=0.6
    pars["diurn_p"]=2.0

    #Initial conditions (mean and standard deviation of the initial CRH field)
    pars["crh_init_mn"]=0.8
    pars["crh_init_sd"]=0.0

    #Experimental configuration
    
    #Total simulated time (days) and timestep (seconds)
    pars["nday"]=1
    pars["dt"]=20.
    
    #Domain size (m) and horizontal resolution (m)
    pars["domain_xy"]=100.e3
    pars["dxy"]=1000.

    #Diagnostics for a netcdf output file with maps
    #Frequency of maps slices (one map every nfig_hr hours)

    pars["nfig_hr"]=1/60

    return(pars)
    
### ALTERNATING-DIRECTION IMPLICIT (ADI) SCHEME FOR A DIFFUSION-REACTION PROBLEM WITH LINEAR SOURCE TERM (see Supporting Information)
#Arguments are a field (fld), the ADI coefficients (a0, a1, a2), the size of the one-dimensional subproblems (nx, ny), an array for the storage of the intermediate solution (x), and the vectors (y, z) associated with the circulant matrices resulting from the ADI discretization. The corresponding linear systems are solved by means of FFT algorithms.  

def ADI(fld, a0, a1, a2, nx, ny, x, y, z):
        f=(1-a0)*fld+a1*np.roll(fld,1,axis=0)+a1*np.roll(fld,-1,axis=0)
        for k in range(ny):
            x[k,:]=solve_circulant(y,f[k,:])
        g=(1-a0-a2)*x+a1*np.roll(x,1,axis=1)+a1*np.roll(x,-1,axis=1)
        for j in range(nx):
            fld[:,j]=solve_circulant(z,g[:,j])
        return(fld)

### MAIN ROUTINE FOR THE 2D MODEL ###

def main(pars):
    execute=sys.argv[0]
    try:
        version=subprocess.check_output(["git", "describe"]).strip()
    except: 
        version="unknown"
    
    #This option allows to test the ADI scheme with initial top-hat CRH profile
    ltest=False
    
    #-------------------
    #Definition of the spatial and temporal discretizations
    #-------------------
    
    #Some options are split from the dictionary to generate the meshes 
    dt=pars["dt"]
    domain_x=domain_y=pars["domain_xy"]
    dxy=pars["dxy"]
    
    #Total number of points in the x and y directions 
    nx=int(domain_x/dxy)+1
    ny=int(domain_y/dxy)+1
    
    #Generation of the spatial numerical mesh, conversion to kilometers
    x1d=np.linspace(0,domain_x,nx)
    y1d=np.linspace(0,domain_y,nx)
    dxkm=dxy/1000.
    X,Y=np.meshgrid(x1d, y1d)
    
    #Temporal discretization 
    nt=int(pars["nday"]*86400/dt)
    times=np.arange(0,nt,1)
    
    #-------------------
    #Auxiliary quantities 
    #-------------------
    
    #Includes all the grid points (will be used in the computation of the distances from any cell to the nearest updraft, = 0 in case the cell itself is developing convection)
    allidx=np.argwhere(np.zeros([nx,ny])<1)
    
    #Flattened array containing a counting of the grid points
    dummy_idx=np.arange(0,nx*ny,1)
    
    #Initialization of the array for convective locations
    cnv_idx=np.zeros([nx,ny],dtype=int)

    #-------------------
    #Definition of some derived quantities
    #-------------------

    #Subsidence velocity is defined as depth of the troposphere (=15 km)/tau_sub
    w_sub=15000./(pars["tau_sub"]*86400.)
    
    #The following quantity is used in the solution of the problem for the convective moistening part
    dt_tau_cnv=dt/pars["tau_cnv"]
    
    #At each time step, some grid cells may be randomly ceased. At the following step, new cells are born to substitute the dying ones. The value cnv_death sets the fixed probability of dying for a cell 
    cnv_death=min(dt/pars["cnv_lifetime"],1.0)
    
    #-------------------
    #Quantities for the numerical solution of the diffusion-reaction problem dR/dt = -D-S with the Alternating Direction Implicit (ADI) scheme  
    #-------------------
    
    #Coefficients (see Supporting Information)
    beta = .5*pars["diffK"]*dt/dxy**2
    alpha = 2*beta
    omega = dt/(2*pars["tau_sub"]*86400.)

    #Initialization and storage of arrays for the solution of the circulant tridiagonal systems resulting from discretization with ADI
    x_crh = np.zeros((ny,nx))
    y_crh = np.zeros(nx)
    y_crh[0] = 1+omega+alpha
    y_crh[1] = -beta
    y_crh[-1] = -beta

    z_crh = np.zeros(nx)
    z_crh[0] = 1+alpha
    z_crh[1] = -beta
    z_crh[-1] = -beta
    
    #-------------------
    #Quantities for the solution of the problem with cold pools included, performed with ADI. The equation is dCIN/dt = -CIN/tau_cin + K∇**2 CIN.    
    #-------------------

    beta_cin=.5*pars["diffCIN"]*dt/dxy**2
    alpha_cin = 2*beta_cin
    omega_cin = .5*dt/pars["tau_cin"]
    x_cin = np.zeros((ny,nx))
    y_cin = np.zeros(nx)
    y_cin[0] = 1+omega_cin+alpha_cin
    y_cin[1] = -beta_cin
    y_cin[-1] = -beta_cin

    z_cin = np.zeros(nx)
    z_cin[0] = 1+alpha_cin
    z_cin[1] = -beta_cin
    z_cin[-1] = -beta_cin
    
    #Sas's doing
    #-------------------
    #Quantities for the solution of the problem with cold pools lifting at the edges

    beta_cape=.5*pars["diffCAPE"]*dt/dxy**2
    alpha_cape = 2*beta_cape
    omega_cape = .5*dt/pars["tau_cape"]
    x_cape = np.zeros((ny,nx))
    y_cape = np.zeros(nx)
    y_cape[0] = 1+omega_cape+alpha_cape
    y_cape[1] = -beta_cape
    y_cape[-1] = -beta_cape

    z_cape = np.zeros(nx)
    z_cape[0] = 1+alpha_cape
    z_cape[1] = -beta_cape
    z_cape[-1] = -beta_cape
    
    #-------------------
    #Netcdf output files
    #-------------------
    
    #Name of the output files
    tab="diffK"+str(pars["diffK"])+"_tausub"+str(pars["tau_sub"])[0:6]+"_crhad"+str(pars["crh_ad"])+"_cin_radius"+str(pars["cin_radius"])+"_downdraft_velocity"+str(pars['downdraft_velocity'])+"_diurn"+str(pars["diurn_opt"])+"_diffCIN"+str(pars["diffCIN"])+'_shear'+str(pars["shear"])
    
    print ("toy_diffusion_2d model of atmosphere")
    print ("opening output maps/stats:",tab)

    #Open the netcdf files (file 1 contains plan views, file 2 contains statistics timeseries)
    nc1=Dataset("td_maps_"+tab+".nc", "w", format="NETCDF4")
    nc2=Dataset("td_stats_"+tab+".nc", "w", format="NETCDF4")
            
    #File 1 (maps)
    time1=nc1.createDimension("time", None)
    var_time1=nc1.createVariable("time","f8",("time",))
    
    x1=nc1.createDimension("x", ny)
    var_x=nc1.createVariable("X","f4",("x",))
    var_x.units="km"
    var_x[:]=x1d
    
    y1=nc1.createDimension("y", nx)
    var_y=nc1.createVariable("Y","f4",("y",))
    var_y.units="km"
    var_y[:]=y1d
    
    var_CRH=nc1.createVariable("CRH","f4",("time","y","x",))
    var_CRH.long_name="Column total water relative humidity"
    var_CRH.units="fraction"
    
    probabilities = nc1.createVariable("PROB","f4",("time","y","x",))
    probabilities.long_name = "Probabilities"
    probabilities.units = 'percentage'

    
    cin_anom_save = nc1.createVariable("CIN anom","f4",("time","y","x",))
    cin_anom_save.long_name = "cin anom"
    cin_anom_save.units = 'cin unit'
    
    var_D2C=nc1.createVariable("D2C","f4",("time","y","x",))
    var_D2C.long_name="Distance to nearest updraft"
    var_D2C.units="km"

    if pars["cin_radius"]>0:
        var_CIN=nc1.createVariable("CIN","f4",("time","y","x",))
        var_CIN.units="fraction"
        
        if pars['downdraft_velocity']>0:
            var_CAPE=nc1.createVariable("CAPE","f4",("time","y","x",))
            var_CAPE.units="fraction"

        cin_grad_save = nc1.createVariable("CIN_grad","f4",("time","y","x",))
        cin_grad_save.long_name = "gradient"
        cin_grad_save.units = 'cin unit'

   
    #Counter for the number of overwritings of the maps file 
    nccnt = 0

    #File 2 (statistics)
    dim_time2 = nc2.createDimension("time", None)
    var_time2 = nc2.createVariable("time","f8",("time",))
    
    crh_mean = nc2.createVariable("CRH_mean","f8",("time",))
    crh_mean.long_name = "CRH domain mean"
    
    crh_std = nc2.createVariable("CRH_std","f8",("time",))
    crh_std.long_name = "CRH domain standard deviation"
    
    crh_in_new = nc2.createVariable("CRH_new_conv","f8",("time",))
    crh_in_new.long_name = "CRH value in new convective locations"
    crh_in_new.units="fraction"
    
    crh_driest=nc2.createVariable("CRH_driest","f8",("time",))
    crh_driest.long_name = "Minimum CRH value"
    crh_driest.units="fraction"
    
    nc2_ncnv=nc2.createVariable("nconv","i8",("time",))
    nc2_ncnv.long_name="Total number of convective events"
    nc2_ncnv.units="total number"
    
    d2c_95=nc2.createVariable("D2C95","f8",("time",))
    d2c_95.long_name="distance to convection 95th percentile"
    d2c_95.units="km"
    
    d2c_max=nc2.createVariable("D2C_max","f8",("time",))
    d2c_max.long_name="distance to convection - maximum"
    d2c_max.units="km"
    
    d2c_mean=nc2.createVariable("D2C_mean","f8",("time",))
    d2c_mean.long_name="distance to convection - mean"
    d2c_mean.units="km"
    
    d2c_median=nc2.createVariable("D2C_median","f8",("time",))
    d2c_median.long_name="distance to convection - median"
    d2c_median.units="km"
    
    var_time1.units=var_time2.units="seconds since 2000-01-01 00:00:00.0"
    var_time1.calendar=var_time2.calendar="gregorian"
    
    #Global attributes for both files
    nc1.description="2D diffusion model, 2d slice snapshots"
    nc2.description="2D diffusion model, Timeseries statistics"
    nc1.history=nc2.history="Created "+datetime.today().strftime('%Y-%m-%d')
    nc1.source=nc2.source="Adrian Tompkins (tompkins@ictp.it)"
    nc1.center=nc2.center="International Centre for Theoretical Physics (ICTP)"
    nc1.exe=nc2.exe=execute
    nc1.version=nc2.version=version

    #All user-defined values in par are also saved as global attributes
    print ("------------")
    print ("- RUN PARS -")
    print ("------------")
    for key,val in pars.items():
        print (key,"=",val)
        setattr(nc1,key,pars[key])
        setattr(nc2,key,pars[key])
    print ("-----------")   

    #-------------------
    #Determining the size of the convective population
    #-------------------

    #Total number of events to distribute, based on a mass conservation argument by Tompkins and Craig (1998), https://doi.org/10.1002/qj.49712455013 
    ncnv_tot=int(nt*nx*ny*w_sub/pars["w_cnv"])

    #3 different options to make the convective population size time-dependent, mimicking diurnal cycle
    if pars["diurn_opt"]==0:
        pdiurn=np.ones(nt)
    if pars["diurn_opt"]==1:
        pdiurn=pars["diurn_a"]*np.sin(np.pi*2*times*dt/86400.)+1.0
    if pars["diurn_opt"]==2:
        pdiurn=(np.sin(np.pi*2*times*dt/86400.)+1.0)**pars["diurn_p"]
    #Normalization to get a probability
    pdiurn/=np.sum(pdiurn)

    #The convective events are distributed among the time steps but temporal variability in their outbreak is accounted for. The population size is not constant but a smoothing filter to lifetime of convection is then applied
    ncnv=np.bincount(np.random.choice(times,ncnv_tot,p=pdiurn),minlength=nt)
    Nsmth=int(pars["cnv_lifetime"]/dt)
    if Nsmth>1:
        ncnv=uniform_filter1d(ncnv,size=Nsmth)
        
    #Storage for overflow. It may happen that too many convective events are still alive, and the number of new events to locate is negative. In this case the routine is designed to borrow events from future time steps and then to even the odds (see below)
    ncnv_overflow=0
    
    #-------------------
    #Initial fields
    #-------------------

    #Initialization of the CRH field
    crh=np.random.normal(loc=pars["crh_init_mn"], scale=pars["crh_init_sd"], size=[ny,nx])

    #Initialization of the CRH field if the ADI solver is to be tested with initial top-hat profile
    if ltest:
        mp=int(nx/2)
        radius=20
        crh[mp-radius:mp+radius,mp-radius:mp+radius]=1.0
    
    #Initialization of the CIN array accounting for the action of coldpool centers
    cin=np.zeros([nx,ny])
    #Sas's doing
    #Initialization of the CIN array accounting for the action of coldpool edges
    cape=np.ones([nx,ny])
    #Initialization of the memory dictionary to save the lifetime information of each convective cell
    memory={}
    cnv_loc=[]

    #Loop over time
    for it in tqdm(range(nt)):
         
        #-------------------
        #Determining the number ncnv_new of NEW convective events
        #-------------------
        
        #Currently active columns
        ncnv_curr=np.sum(cnv_idx)
        
        ncnv_new=ncnv[it]+ncnv_overflow-ncnv_curr
        ncnv_overflow=0             
        if (ncnv_new<0):
            ncnv_overflow=ncnv_new                            
            ncnv_new=0
        
        #-------------------
        #Choice of the new convective locations according to the PDF derived from Bretherton et al. (2004)'s relationship
        #-------------------
        
        prob_crh=np.exp(pars["crh_ad"]*crh)
        
        #The cells already developing convection are excluded                                                                                                                                     
        prob_crh*=(1-cnv_idx)
         
        prob_crh/=np.mean(prob_crh)     
        prob=prob_crh     
        
        #Account for convective inhibiting action of coldpool centers
        prob_cin=1.0-cin
        if pars["cin_radius"]>0:                                                                                                                                                                           
            prob*=prob_cin
            #Account for convective initiation action of coldpool edges
            if pars['downdraft_velocity']>0:
                prob*=cape                                                                                                                                                    
        
        #Normalization to get a PDF
        prob/=np.sum(prob)                                                                                                                                                               
        prob1d=prob.flatten()
        
        #A number ncnv_new of new events are placed according to the PDF and the corresponding grid points are assigned value 1 for the indicator function                                                                                                                                                                    
        coords=np.random.choice(dummy_idx,int(ncnv_new),p=prob1d,replace=False)
        new_loc=np.unravel_index(coords,(nx,ny))
        cnv_idx[new_loc]=1                                                                                                                                                  

        #-------------------
        #Numerical solution with Strang splitting
        #-------------------
        
        #Analytical solution of the problem dR/dt = C for half a time step 
        if not ltest:
            crh = (pars["crh_det"]+(crh-pars["crh_det"])*np.exp(-0.5*dt_tau_cnv))*cnv_idx+crh*(1-cnv_idx)

        #Solution of the problem dR/dt = -D-S with ADI method for a full time step
        crh = ADI(crh, alpha, beta, omega, nx, ny, x_crh, y_crh, z_crh)

        #Analytical solution of dR/dt = C for half a time step 
        if not ltest:
            crh = (pars["crh_det"]+(crh-pars["crh_det"])*np.exp(-0.5*dt_tau_cnv))*cnv_idx+crh*(1-cnv_idx)

        #-------------------
        #Routine to calculate the distance to the nearest convective updraft (Probably redundant now!)
        #-------------------
        
        cnv_coords=np.argwhere(cnv_idx)
        ncnv_curr=np.sum(cnv_idx)
        nc2_ncnv[it]=ncnv_curr
        
        #The procedure takes into account the doubly-periodic nature of the computational domain
        if ncnv_curr>0:
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
            cnvdst*=dxkm
        else:
            cnvdst=np.ones([nx,ny])*1.e6

        #Account for cold pool inhibition effect 
        if pars["cin_radius"]>0:
            # maskcin=np.where(cnvdst<pars["cin_radius"],1,0)
            maskcin=create_time_dependent_mask(cnv_loc, memory, pars['cin_radius'], pars['downdraft_velocity'], pars["cp_init_delay"], nx, ny, dxkm, dt)[0]
            # the mask is shifted due to wind shear effect
            shifted_maskcin1=np.roll(maskcin,int(pars["shear"]*1000/dxy),axis=0)
            
            #cin = 1 for the points within distance cin_radius from the convective source
            cin=cin+shifted_maskcin1 
            cin=np.clip(cin,0,1)
                   
            #Solution of the differential problem for CIN with ADI method            
            cin = ADI(cin, alpha_cin, beta_cin, omega_cin, nx, ny, x_cin, y_cin, z_cin)
            
            #Some diagnostic variables that may be be useful
            cin_grad=np.sqrt(np.nansum(np.array(np.gradient(cin))**2,axis=0)) #Sas's doing
            cin_anom=cin-np.nanmean(cin)
            
            ########
            cin=np.clip(cin,0,1)
            
            #This is Sas's doing
            # Account for convective initiation at the cold pool edge
            if pars['downdraft_velocity']>0:
                #create a cape mask considering the lifetime of the convective cells and gust velocity
                maskcin1=create_time_dependent_mask(cnv_loc, memory, pars['cin_radius'], pars['downdraft_velocity'], pars["cp_init_delay"], nx, ny, dxkm, dt)[1]
                #the mask is shifted due to the effect of wind shear
                shifted_maskcin2=np.roll(maskcin1,int(pars["shear"]*1000/dxy),axis=0)
                #the edge of the mask is converted to 1 and the rest to 0
                maskcape=find_0_nearby_1(shifted_maskcin2)
                #CAPE=1 for the points at the edge of the cold pool
                cape=cape+maskcape
                cape=np.clip(cape,0,1)
                
                #Solution of the differential problem for CAPE with ADI method   
                cape = ADI(cape, alpha_cape, beta_cape, omega_cape, nx, ny, x_cape, y_cape, z_cape)
                cape=np.clip(cape,0,1)
            
        #At each time step, some convective cells may be randomly killed
        mask=np.where(np.random.uniform(size=(ny,nx))<=cnv_death,0,1)
        cnv_idx*=mask
        cnv_loc=np.argwhere(cnv_idx==1)
        
        #Sas's doing
        #Create memory
        for loc in cnv_loc:
            if str(loc) in memory:
                memory[str(loc)]+=1
            else:
                memory[str(loc)]=1
        #Delete non existing cells from memory
        temp_loc=memory.copy()
        for loc in temp_loc.keys():
            if loc not in [str(x) for x in cnv_loc]:
                memory.pop(str(loc))
                
        #-------------------
        #Writing of netcdf output
        #-------------------
 
        #At each time step for timeseries
        var_time2[it]=it*dt
        crh_mean[it]=np.mean(crh)
        crh_std[it]=np.std(crh)
        if ncnv_new>0:
            crh_in_new[it]=np.mean(crh[new_loc])
        crh_driest[it]=np.min(crh)
        
        d2c_mean[it]=np.mean(cnvdst)
        d2c_median[it]=np.median(cnvdst)
        d2c_max[it]=np.max(cnvdst)
        d2c_95[it]=np.percentile(cnvdst,95)
        
        #Every nfig_hr hours for maps
        if it%int(pars["nfig_hr"]*3600/dt)==0:
            var_time1[nccnt]=it*dt
            var_CRH[nccnt,:,:]=crh     
            var_D2C[nccnt,:,:]=cnvdst
            probabilities[nccnt,:,:]=prob   
            if pars["cin_radius"]>0:
                var_CIN[nccnt,:,:]=cin  
                if pars['downdraft_velocity']>0:
                    var_CAPE[nccnt,:,:]=cape     
                cin_grad_save[nccnt,:,:]=cin_grad
                cin_anom_save[nccnt,:,:]=cin_anom
            nccnt+=1

    nc1.close()
    nc2.close()


if __name__ == "__main__":
    pars=defaults() 
    pars=getargs.getargs(pars)
    main(pars)
