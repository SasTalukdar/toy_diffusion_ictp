#!/bin/python                                                                                       

import os
from glob import glob
import subprocess as subproc
from multiprocessing import Pool,cpu_count
import getopt, sys
import toy_diffusion_2d
import ast

def main(argv):

    lparallel=True

    f=open("diffusion_results.txt","w")
    f.close()

    #Default values are listed in toy_diffusion_2d.py
    pars=toy_diffusion_2d.defaults()
    
    odir="./"

    arglist=["help","diffK=","crh_ad=","diurn_opt=","tau_sub=","odir=","nfig_hr=","cin_radius=","dt=","nday=","crh_init_mn=","domain_xy=","dxy="]
    try:
        opts, args = getopt.getopt(argv,"h",arglist)
    except getopt.GetoptError:
        print ("arg error")
        print (argv)
   
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h","--help"):
            print ("check out this list: ",arglist)
            sys.exit()
        elif opt in ("--nday"):
            pars["nday"] = int(arg)
        elif opt in ("--dt"):
            pars["dt"] = float(arg)
        elif opt in ("--diffK"):
            list_diffK = ast.literal_eval(arg)
        elif opt in ("--tau_sub"):
            list_tau_sub = ast.literal_eval(arg)
        elif opt in ("--crh_ad"):
            list_crh_ad = ast.literal_eval(arg)
        elif opt in ("--cin_radius"):
            list_cin_radius = ast.literal_eval(arg)
        elif opt in ("--crh_init_mn"):
            pars["crh_init_mn"] = float(arg)
        elif opt in ("--domain_xy"):
            pars["domain_xy"] = float(arg)
        elif opt in ("--dxy"):
            pars["dxy"] = float(arg)
        elif opt in ("--diurn_opt"):
            list_diurn_opt = ast.literal_eval(arg)
        elif opt in ("--odir"):
            pars["odir"] = arg

    #Make a list of dictionaries with all the combinations of the five  parameters diffK, tau_sub, a_d, cin_radius, diurn_opt

    arglist=[{**pars,"diffK":d,"tau_sub":t,"crh_ad":c,"cin_radius":cr,"diurn_opt":dc} for d in list_diffK for t in list_tau_sub for c in list_crh_ad for cr in list_cin_radius for dc in list_diurn_opt]

    print("check ",arglist)

    if (lparallel): 
    #parallel mode
        ncore=min(len(arglist),int(cpu_count()))
        os.system('taskset -cp 0-%d %s' % (ncore, os.getpid()))
        with Pool(processes=ncore) as p:
            p.map(toy_diffusion_2d.main,arglist)
        print ("done parallel")
    else: 
    #serial mode
        for args in arglist:
             toy_diffusion_2d.main(args)
        print ("done serial")

if __name__ == "__main__":
    main(sys.argv[1:])
