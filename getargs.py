#!/usr/bin/python

import argparse

def getargs(defaultpars):
    parser = argparse.ArgumentParser(description='Dynamic arguments')
    for key,val in defaultpars.items():
        parser.add_argument('--'+key,type=type(val))
    newpars=vars(parser.parse_args())
    for key,val in newpars.items():
        if val==None:
            newpars[key]=defaultpars[key]
    return(newpars)




