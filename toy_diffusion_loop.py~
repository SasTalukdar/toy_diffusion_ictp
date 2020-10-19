#!/bin/python                                                                                       

import os
from glob import glob
import subprocess as subproc
from multiprocessing import Pool,cpu_count
import getopt, sys
import toy_diffusion_2d
import ast

def main(argv):
    """ entry point"""

    lparallel=True

    f=open("diffusion_results.txt","w")
    f.close()

    # defaults here as LISTS!
    diffK=[37500]
    crh_ad=[16.12]
    tau_sub=[20]
    cin_radius=[-99] # turned off by default
    diurn_cases=[0] # default value for all runs

    # other defaults
    nfig_hr=6
    odir="./"

    arglist=["help","diffK=","crh_ad=","tau_sub=","odir=","nfig_hr=","cin_radius"]
    try:
        opts, args = getopt.getopt(argv,"h",arglist)
    except getopt.GetoptError:
        print (argv)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h","--help"):
            print ("check out this list: ",arglist)
            sys.exit()
        elif opt in ("--diffK"):
            diffK = ast.literal_eval(arg)
        elif opt in ("--crh_ad"):
            crh_ad = ast.literal_eval(arg)
        elif opt in ("--tau_sub"):
            tau_sub = ast.literal_eval(arg)
        elif opt in ("--cin_radius"):
            cin_radius = ast.literal_eval(arg)
        elif opt in ("--nfig_hr"):
            nfig_hr = int(arg)
        elif opt in ("--odir"):
            odir = arg

    # make a list of dictionaries with ALL combinations of the 3 arguments
    arglist=[{"diffK":d,"tau_sub":t,"crh_ad":c,"nfig_hr":nfig_hr,"odir":odir,"cin_radius":cr,"diurn_cases":dc} for d in diffK for t in tau_sub for c in crh_ad for cr in cin_radius for dc in diurn_cases ]    
    #

    print("check ",arglist)

    # now farm out the jobs over the triple loop
    # only use the number of processors needed, or max-1
    if (lparallel): # parallel mode
        ncore=min(len(arglist),int(cpu_count()))
        #os.sched_setaffinity(0, set(range(cpu_count())))
        os.system('taskset -cp 0-%d %s' % (ncore, os.getpid()))
        with Pool(processes=ncore) as p:
            p.map(toy_diffusion_2d.main,arglist)
        print ("done")
    else: # serial model
        for args in arglist:
             toy_diffusion_2d.main(args)

if __name__ == "__main__":
    main(sys.argv[1:])
