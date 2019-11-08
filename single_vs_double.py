#!/usr/bin/env python

import os, sys
import optparse

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("-t", "--type",default="double")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not opts.type in ["single","double"]:
    print("Only single or double precision supported")
    exit(0)

gcepyx = "gcex/GCE.pyx"
gcecpp = "gcex/GCE.cpp"
globalh = "gcex/src/global.h"
gcepy = "gcex/gce.py"
kernelcu = "gcex/src/kernel.cu"

f = open(gcepyx,'r')
filedata = f.read()
f.close()

if opts.type == "single":
    newdata = filedata.replace("float64","float32")
else:
    newdata = filedata.replace("float32","float64")

f = open(gcepyx,'w')
f.write(newdata)
f.close()

f = open(gcecpp,'r')
filedata = f.read()
f.close()

if opts.type == "single":
    newdata = filedata.replace("float64","float32")
else:
    newdata = filedata.replace("float32","float64")

f = open(gcecpp,'w')
f.write(newdata)
f.close()

f = open(globalh,'r')
filedata = f.read()
f.close()

if opts.type == "single":
    newdata = filedata.replace("double","float")
else:
    newdata = filedata.replace("float","double")    

f = open(globalh,'w')
f.write(newdata)
f.close()

f = open(gcepy,'r')
filedata = f.read()
f.close()

if opts.type == "single":
    newdata = filedata.replace("float64","float32")
else:
    newdata = filedata.replace("float32","float64")     

f = open(gcepy,'w')
f.write(newdata)
f.close()

f = open(kernelcu,'r')
filedata = f.read()
f.close()

if opts.type == "single":
    newdata = filedata.replace(" fmod("," fmodf(")
else:
    newdata = filedata.replace(" fmodf("," fmod(")

f = open(kernelcu,'w')
f.write(newdata)
f.close()
