#!/bin/python

# this uses SLURM dependencies to loop over dates
# we could submit each model and rcp here too, using the serial queue 
# but that will be slow due to limit of runnign jobs per person
# so i will farm out the model and rcp to another python script which splits the tasks


import os, sys, getopt
import subprocess
from glob import glob
#srcdir="/home/netapp-clima/users/tompkins/ISIMIP2/"

def main(argv):
    """ entry point"""

    # send email to user:
    email=os.environ["USER"]+"@ictp.it"

    # get the queue string
    try:
        opts, args = getopt.getopt(argv,"h",["queue="])
    except getopt.GetoptError:
        print (argv)
        sys.exit(2)

    queue="esp1" # default queue

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
        ncore=20

    # make the vectors of values here
    diffK=[1.3**i for i in range(27,43)]

    tau_sub=[i for i in range(4,44,2)]
    # chunk the tau_sub array
    nchunk=6 # 3 runs at a time?
    tau_sub=[tau_sub[i:i+nchunk] for i in range(0,len(tau_sub),nchunk)]

    #
    #crh_ad=[10,12,14,18,20,22]+[14.72,16.12]
    # short test run for long queue (max 6 jobs)
    #crh_ad=[10,12,14,16,18,20]

    # test single job only
    crh_ad=[16.12]

    # will make ONE job for each crh_ad and these loop over the other two variables.
    for itau,taulist in enumerate(tau_sub):
        for icrh,crh in enumerate(crh_ad):
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
                fh.writelines('python3 ~/diffusion/toy_diffusion/toy_diffusion_loop.py --diffK="{}" --tau_sub="{}" --crh_ad="{}" \n'.format(diffK,taulist,[crh]))

        #jobid=os.system("sbatch "+jobfile)
        command=["sbatch","--parsable",jobfile]
        jobid=subprocess.check_output(command)
        jobid=jobid.decode('utf-8')
        print("submitted ",jobid)

if __name__ == "__main__":
    main(sys.argv[1:])

