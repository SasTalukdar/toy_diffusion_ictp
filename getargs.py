#!/usr/bin/python

import argparse

def getargs(defaultpars):
    parser = argparse.ArgumentParser(description='Dynamic arguments')
    namespace = argparse.Namespace(**defaultpars)
    # add each key of the default dictionary as an argument
    # expecting the same type 
    for key, val in defaultpars.items():
        try:
            typ = type(val[0])
            nargs = "+"
        except TypeError:
            typ = type(val)
            nargs = None
        parser.add_argument('--'+key, nargs=nargs, type=typ)
        #parser.add_argument('--'+key, type=type(val))
    parser.parse_args(namespace=namespace)
    return vars(namespace)

