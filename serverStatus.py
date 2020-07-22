# Program to get current status of astro computing machines
# written by Matthew Smith June 2020

# import modules
import argparse
import parser
import os
from os.path import join as pj
import pickle
import sys


# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--getRemoteInfo", help="Remote info gatherer activation",
                    action="store_true")
parser.add_argument("-n", "--noWindow", help="Run without X-windows", action="store_true")
parser.add_argument('-i', '--infoCollect', nargs='+', default=[])
# get arguments
args = parser.parse_args()

##########################################################################
##########################################################################
##########################################################################

# set default parameters

# list of machines
defaultMachines = ["herschel01", "caroline" "scoobydoo", "serpens"]

##########################################################################
##########################################################################
##########################################################################


# remote info arg
if args.getRemoteInfo:
    getRemoteInfo = True
else:
    getRemoteInfo = False
# x-window param
if args.noWindow:
    xWin = False
else:
    xWin = True

sys.path.append("/home/gandalf/spxmws/Hard-Drive/scripts/python/Misc")
from serverStatusModules import *


# see if home folder created, to store preferences, and temporary information
homeDir = os.getenv("HOME")
if homeDir is None:
    homeDir = os.getenv("home")
if homeDir is None:
    raise Exception("Unable to locate home directory")
if os.path.isdir(pj(homeDir,".serverStatus")) is False:
    os.mkdir(pj(homeDir,".serverStatus"))
#serverFile = pj(homeDir,".serverStatus","serverInfo.pkl")
scriptPath = os.path.realpath(__file__)
serverFile = pj(scriptPath[0:-len(scriptPath.split("/")[-1])], "serverInfo.pkl")


# see if in gathering mode or standard
if getRemoteInfo:
    # get what information to gather from command line input
    if len(args.infoCollect) > 0:
        infoToGather = {}
        tempInfo = args.infoCollect
        for i in range(0,len(tempInfo)):
            splitInfo = tempInfo[i].split(":")
            infoToGather[splitInfo[0]] = bool(splitInfo[1])
    else:
        infoToGather = {"boot":True, "CPU":True, "memory":True, "disk":True, "network":True, "GPU":True, "misc":True}
    
    gatherComputerInfo(serverFile, infoToGather)
else:
    # see if a list of default machines exists
    defaultFile = pj(homeDir,".serverStatus","defaultMachine.pkl")
    if os.path.isfile(defaultFile):
        fileIn = open(defaultFile,'rb')
        defaultMachines = pickle.load(fileIn)
        fileIn.close()
    else:
        defaultMachines = ['herschel01', 'caroline', "serpens"]

    serverStatusMain(serverFile, xWin, defaultMachines, scriptPath, defaultFile)
    
    print("Program Finished Successfully - Have a nice day!")
    