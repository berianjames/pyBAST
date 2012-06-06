#!/usr/bin/env python
# encoding: utf-8
"""
This file is part of pyBAST, Bayesian Astrometry
Copyright (C) Joshua S. Bloom. All Rights Reserved. 2012.

See the license file as part of this project on github.

Code:
https://github.com/berianjames/pyBAST
"""

import sys
import os, copy
import pyBA
import sdss
import numpy as np
import cPickle as pickle

class WD(object):
    """
    A worked example measuring the proper motion of a white dwarf on the sky.
    """
    
    def __init__(self,pos=(342.1913750, -0.9019444)):
        "WD @ 22:48:45.930 -00:54:07.00 J2000 "
        " 224845.93−005407.0 pm = 204±5 mas/yr"
        self.pos = pos
        
    def run(self,test=True):
        self.prepare(runids="all" if not test else [2662,4198,7195])
        self.fit_all()
        self.locate_source()
        
    def prepare(self,runids=[2662,4198,7195]):
        """
        get's the master catalog around the position and generates match files
        """
        self.a = sdss.sdssq()
        ## this will make a query to the Stripe82 catalog
        self.master = self.a.get_master_cat(pos=self.pos) ## this can take awhile

        ## this will generate the match files from the master catalog
        self.a.write_match_files(self.master,runids=runids)

        ## self.a.generated_match_files contains the paths now to the match files
        ##   along with the number of matches and the time of the observations
    
    def fit_all(self,save_output=True,clobber=False,minmatch=20,mymatches=[], \
                sigma_reject=5.0,verbose=True):
        """
        runs through all the match files and fits the astrometry
        
        clobber -- overwrite output file if it exists
        save_output -- save the output file
        minmatch  -- minimum number of required matches to perform the mapping
        mymatches -- list of files to match (instead of what gets populated in prepare)
        sigma_reject -- rejection of outliers [not used for now!]
        """
        if mymatches != []:
            matches = mymatches
        else:
            matches = self.a.generated_match_files
        
        self.pybast_files = []
        
        for fname,nmatch,t in matches:
            
            if nmatch < minmatch:
                print "skipping %s ... too few matches [%i]" % (fname,nmatch)
                continue
            
            print "*"*60
            print "Working on %s" % (fname,)
            sys.stdout.flush() 
            
            outname = "".join(fname.split(".")[0:-1]) + ".pyBAST"
            
            if not clobber and os.path.exists(outname):
                print " ... skipping (mapping file exists)"
                self.pybast_files.append((outname,fname,t))    
                continue
                
            data = sdss.csv2rec(fname)
            # Parse data array into objects
            # the A objects will be the fiducial positions from the deep coadd
            # the B objects will be source positions in this epoch
            # both are given in arcseconds relative to a fiducial center
            objectsA = np.array( [ pyBA.Bivarg(mu=[x["master_dra"],x["master_ddec"]],\
                                   sigma=np.array([x["master_raerr"],x["master_decerr"]])) \
                                   for x in data ] )
            objectsB = np.array( [ pyBA.Bivarg(mu=[x["dra"],x["ddec"]], \
                                   sigma=np.array([x["raerr"],x["decerr"]])) \
                                   for x in data ] )
            
            # Suggest starting point for background mapping
            S = pyBA.background.suggest_mapping(objectsA,objectsB)

            # Get maximum a posteriori background mapping parameters
            P = pyBA.background.MAP( objectsA, objectsB, mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True )
            D = pyBA.Amap(P,objectsA, objectsB)
            if verbose:
                print " ... conditioning"
                sys.stdout.flush()
            D.condition()
            if verbose:
                print D.hyperparams
                                   
            if save_output:
                if clobber and os.path.exists(outname):
                    os.remove(outname)
                
                output = open(outname, 'wb')
                pickle.dump(D, output, protocol=-1)
                if verbose:
                    print "   ... wrote %s" % outname
                self.pybast_files.append((outname,fname,t))
                
        
    def locate_source(self,mypybastfiles=[]):

        if mypybastfiles != []:
            pybastf = mypybastfiles
        else:
            pybastf = self.pybast_files
        
        rez = {}    
        for pyfile,fname,t in pybastf:
            ## locate the position of the source
            if not os.path.exists(fname) or not os.path.exists(pyfile):
                print "Cannot locate the source for %s. Needed files not found." % fname
                continue
            f = open(fname,"r") ; header = [l[1:].strip() for l in f.readlines() if l[0] == "#"] ; f.close()
            epoch_info = {}
            for h in header:
                if h.find(":") != -1:
                    ttt = h.split(":")
                    if ttt[1].find(",") != -1:
                        v = tuple([float(x) for x in ttt[1].split(",")])
                    else:
                        v = ttt[1].strip()
                    epoch_info[ttt[0].strip()] = v
            rez[epoch_info["filename"]] = copy.copy(epoch_info)
        

            if not epoch_info.get("converted source location (delta ra, delta dec)"):
                print "Source not found in file %s" % (epoch_info["filename"],)
                continue
            print "-"*60
            pos = epoch_info["converted source location (delta ra, delta dec)"]
            err = epoch_info["source error (raerr, decerr)"]
            
            print "Working on source in %s at position %s (t=%s day)" % \
                  (epoch_info["filename"],pos,epoch_info['observation time (day)'])
            
            mapfile = open(pyfile, 'rb')
            D = pickle.load(mapfile)
            mapfile.close()
            pos = epoch_info["converted source location (delta ra, delta dec)"]
            BV = pyBA.Bivarg(mu=pos,sigma=np.array([err[0],err[1]]))
            R = D.regression(BV)
            print R[0]
            rez[epoch_info["filename"]]["pos"] = R
        
        self.rez = rez
    
    def gen_results(self):
        ## TODO:
        ##   make a plot of RA/DEC v. time
        ##   make a plot of the location on the sky (including error ellipses)
        ##   calculate the proper motion
        pass
           
def main(test=True):
    w = WD()
    w.run(test=test)
    return w


if __name__ == '__main__':
    main()

