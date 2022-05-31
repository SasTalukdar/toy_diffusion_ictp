#!/bin/python

#SLURM dependencies are used to loop over data

import os, sys, getopt
import subprocess
from glob import glob

def main(argv):

    #An email is sent to the user
    email=os.environ["USER"]+"@ictp.it"

    #Get the queue string (default is esp1)
    try:
        opts, args = getopt.getopt(argv,"h",["queue="])
    except getopt.GetoptError:
        print (argv)
        sys.exit(2)
    queue="esp1"
    
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print("pass the queue string")
            sys.exit()
        elif opt in ("--queue"):
            queue=arg

    if queue=="esp1":
        ncore=20
    if queue=="esp":
        ncore=12
    if queue=="long":
        ncore=12

    #Creation of the vectors of values
    diffK=[5000,7000,10000,15000,20000]

    tau_sub=[i for i in range(5,30,5)]

    #The vectors tau_sub and a_d can be conveniently chunked to optimize the use of single nodes
    nchunk=1
    tau_sub=[tau_sub[i:i+nchunk] for i in range(0,len(tau_sub),nchunk)]

    crh_ad=[14.72]
    nchunk_ad=1
    crh_ad=[crh_ad[i:i+nchunk_ad] for i in range(0,len(crh_ad),nchunk_ad)]
    
    diurn_opt=[0]
    
    cin_radius=[-99]

    #Loop over the variables
    for itau,taulist in enumerate(tau_sub):
        for icrh,crh in enumerate(crh_ad):
            for icr,cr in enumerate(cin_radius):
                jobfile="dif_"+str(itau)+"_"+str(icrh)
                with open(jobfile,"w") as fh:
                    jobname=jobfile
                    fh.writelines("#!/bin/bash\n")
                    fh.writelines("#SBATCH --job-name=%s\n" % jobname)
                    fh.writelines("#SBATCH -p {}\n".format(queue))
                    fh.writelines("#SBATCH -N 1 --ntasks-per-node={}\n".format(ncore))
                    fh.writelines("#SBATCH -t 0-24:00\n")     
                    fh.writelines("#SBATCH -o ./output/slurm.%j.out\n")
                    fh.writelines("#SBATCH -e ./output/slurm.%j.err\n")
                    fh.writelines("#SBATCH --mail-type=ALL\n")
                    fh.writelines("#SBATCH --mail-user={}\n".format(email))
                    fh.writelines("cd $SLURM_SUBMIT_DIR\n")
                    fh.writelines("source /opt-ictp/ESMF/env201906\n")
                    fh.writelines("export NETCDF_LIB=$(nf-config --flibs)\n")
                    fh.writelines("export NETCDF_INCLUDE=$(nf-config --fflags)\n")
                    fh.writelines("export FC=`nf-config --fc`\n")
                    fh.writelines('python3 ~/diffusion/toy_diffusion/toy_diffusion_loop.py --nday={} --dt="{}" --diffK="{}" --tau_sub="{}" --crh_ad="{}" --crh_init_mn="{}" --domain_xy="{}" --dxy="{}" --diurn_opt="{}" --cin_radius="{}" \n'.format(180,30,diffK,taulist,crh, 0.8, 300e3, 2e3, diurn_opt,[cr]))

                command=["sbatch","--parsable",jobfile]
                jobid=subprocess.check_output(command)
                jobid=jobid.decode('utf-8')
                print("submitted ",jobid)

if __name__ == "__main__":
    main(sys.argv[1:])

