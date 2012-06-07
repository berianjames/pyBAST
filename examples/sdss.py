#!/usr/bin/env python

"""
This file is part of pyBAST, Bayesian Astrometry
Copyright (C) Joshua S. Bloom. All Rights Reserved. 2012.

See the license file as part of this project on github.

usage: see the associated ipython notebook in this folder.

Code:
https://github.com/berianjames/pyBAST
"""

import os, sys, string,urllib2,urllib,copy
from math import log10, radians, pi, sin, cos, atan2, sqrt
import time, traceback, datetime
import threading
import StringIO
import numpy as np
from matplotlib.mlab import csv2rec

class sdssq(object):
    """
    query object for Stripe82
    """
    #dr_url="http://cas.sdss.org/astrodr7/en/tools/search/x_sql.asp"
    dr_url="http://cas.sdss.org/stripe82/en/tools/search/x_sql.asp"
    formats = ['csv','xml','html']
    def_fmt = "csv"
    sdss_coadd_platescale = 0.396127      ## arcsec/pix
    def_t0                = 5.21972623E4  # fidual start time (days)
    
    def _filtercomment(self,sql):
        "Get rid of comments starting with --. Function from Tomas Budavari's code (sqlcl.py)"
        fsql = ''
        for line in sql.split('\n'):
            fsql += line.split('--')[0] + ' ' + os.linesep
        return fsql
    
    def recquery(self,sql,url=dr_url,fmt=def_fmt):
        """
        makes a recarray of the query results
        """
        rez = self._query(sql,url=url,fmt=fmt)
        #print rez.readlines()
        #rez.seek(0)
        tmp = rez.readline()
        rez.seek(0)
        if len(tmp) == 0 or tmp.find("error_message") != -1 or tmp.find("ERROR") != -1:
            print "rez:"
            print rez.readlines()
            return np.zeros((1,)).view(np.recarray)
       
        try:
            return csv2rec(rez)
        except:
            print "err"
            print rez.readlines()
            
    def _query(self,sql,url=dr_url,fmt=def_fmt,verbose=False):
        "Run query and return file object"

        fsql = self._filtercomment(sql)
        if verbose:
            print fsql
        params = urllib.urlencode({'cmd': fsql, 'format': fmt})
        try:
                return StringIO.StringIO(urllib2.urlopen(url+'?%s' % params).read())
        except:
                print "TRIED: " + url+'?%s' % params
                print "EXCEPT: sdss.py._query()"
                return StringIO.StringIO() # This is an empty filehandler
    
    def _old_get_master_cat(self,pos=(342.19156468,-0.90203402),errdeg=0.05,rcut=22.5):
        ## I'm not using the full HTM here
        ramin = pos[0] - 0.5*errdeg/np.cos(pos[1]) ; ramax = pos[0] + 0.5*errdeg/np.cos(pos[1]) 
        decmin = pos[1] - 0.5*errdeg ; decmax = pos[1] + 0.5*errdeg
        rc = "and p.r < %f" % rcut if rcut not in [None] else ""
        q = """SELECT p.objid,p.ra,p.dec,p.rowc,p.rowcErr,p.colc,p.colcErr,
                      p.u,p.g,p.r,p.i,p.z,
                      p.run,p.rerun,p.camcol,p.field
               FROM dbo.fGetObjFromRect(%f,%f,%f,%f) n
               join PhotoObjAll p on n.objID=p.objID
               WHERE 
                n.run in (106,206)
                %s
        """ % (ramin,ramax,decmin,decmax, rc)
        

    def dist(self,lon0, lat0, lon, lat):
        """
        Calculates the distance between two points (decimal)
        """
        d_lat = radians(lat0 - lat)
        d_lon = radians(lon0 - lon)
        x = sin(d_lat/2.) ** 2 + \
          cos(radians(lat0)) * cos(radians(lat)) *\
          sin(d_lon/2.) ** 2
        y = 2.0 * atan2(sqrt(x), sqrt(1.0 - x))
        distance = y*180.0/pi
        return distance

    def write_match_files(self,master,pos=(342.1913750,-0.9019444), scale=sdss_coadd_platescale,\
        runids=[4198],t0=def_t0):
        
        self.generated_match_files = []
        
        if isinstance(runids,str):
            if runids == "all":
                runids = list(set(master.run))
            else:
                print "I dont understnd runids = %s" % repr(runids)
                return
        
        ## figure out the special ID of the source we're interested in
        mind = 100.0
        bestid = -1
        gotit = False
        mpos = tuple()
        for x in master:
            d = self.dist(pos[0],pos[1],x["ra"],x["dec"])
            if d < mind:
                mind = d
                bestid = x["master_objid"]
                mpos = (x["master_ra"],x["master_dec"])
            if mind < 1.5/(3600.0):
                gotit= True
                break
        
        if gotit:
            print "source of interest: mind=%.3f arcsec sourceid=%i" % (mind*3600, bestid)
            print " master position = %s" % repr(mpos)
        else:
            print "couldn't find the source after %i sources searched" % len(master)
        
        ## make conversion table for RA,DEC
        convra = lambda ra: (ra - pos[0])*np.cos(pos[1])*3600.0
        convdec = lambda dec: (dec - pos[1])*3600.0
            
        for r in runids:
            ## get only the matches for this run
            tmp = master[np.where(master.run == r)]
            fname = "match_astrom_%.4f_%.5f_run%i.dat" % (pos[0],pos[1],r)
            f = open(fname,"w")
            f.write("# filename: %s \n" % fname)
            f.write("# nominal center (used for x,y conversion): %f, %f\n" % (pos[0],pos[1]))
            if gotit:
                f.write("# source of interest\n")
                f.write("#   master_objid: %i\n#    master_pos: %f, %f\n" % \
                        (bestid,mpos[0],mpos[1]))
                ttt = np.where(tmp["master_objid"] == bestid)
                f.write("#   fiducal master location (delta ra, delta dec): %f, %f\n" % \
                        (convra(mpos[0]),convdec(mpos[1])))
                if len(ttt[0]) > 0:
                    f.write("# converted source location (delta ra, delta dec): %f, %f\n" % \
                            (convra(tmp[ttt]["ra"]),convdec(tmp[ttt]["dec"])))
                    f.write("# source location (ra, dec): %f, %f\n" % \
                                    (tmp[ttt]["ra"],tmp[ttt]["dec"]))
                    f.write("# source error (raerr, decerr): %f, %f\n" % \
                                (tmp[ttt]["raerr"],tmp[ttt]["decerr"]))                
            else:
                f.write("# could not find master source of interest\n")
                
            f.write("# observation time (day): %f\n# t0 (day): %f\n" % (tmp["time"][0],t0))
            f.write("master_objid,objid,master_ra,master_dec,master_dra,master_ddec")
            f.write(",master_raerr,master_decerr,master_radeccov,ra,dec,dra,ddec,raerr,decerr,radeccov")
            f.write(",master_rmag,rmag\n")
            for x in tmp[np.where(tmp["master_objid"] != bestid)]:
                f.write("%i,%i" % (x["master_objid"],x["objid"]))
                f.write(",%f,%f,%f,%f" % (x["master_ra"],x["master_dec"],\
                         convra(x["master_ra"]),convdec(x["master_dec"])))
                f.write(",%f,%f,0.0" % (x["master_raerr"],x["master_decerr"]))
                f.write(",%f,%f,%f,%f" % (x["ra"],x["dec"],\
                         convra(x["ra"]),convdec(x["dec"])))
                f.write(",%f,%f,0.0" % (x["raerr"],x["decerr"]))
                f.write(",%f,%f\n" % (x["master_rmag"],x["rmag"]))
                
            f.close()
            print "wrote %i matches in file %s" % (len(tmp[np.where(tmp["master_objid"] != bestid)]),fname)
            self.generated_match_files.append((fname,len(tmp[np.where(tmp["master_objid"] != bestid)]),tmp["time"][0]))
            
        ## sort the generated file list by time
        self.generated_match_files.sort(key=lambda x: x[2])
        
    def get_master_cat(self,pos=(342.1913750, -0.9019444),errdeg=0.07,rcut=22.5,t0=def_t0,\
        MASTER_PRE = "master_sdss",savefile=True):
        """
        issue the monsterous nested query to get master catalog positions in Stripe82 over time
        """
        master_name = MASTER_PRE + "%.4f_%.5f.npz" % pos

        if os.path.exists(master_name):
            npzfile = np.load(master_name)
            return npzfile["master"].view(np.recarray)
        
        ramin = pos[0] - 0.5*errdeg/np.cos(pos[1]) ; ramax = pos[0] + 0.5*errdeg/np.cos(pos[1]) 
        decmin = pos[1] - 0.5*errdeg ; decmax = pos[1] + 0.5*errdeg
        rc = "and p.r < %f" % rcut if rcut not in [None] else ""
        
        ### Runs 106 and 206 are the deep coadds from Stripe82. Get the master catalog 
        ###    about the input position
        q = """SELECT cc.rr0objid as master_objid,cc.rr0ra as master_ra,cc.qdec as master_dec,
               cc.master_decerr,cc.master_raerr,cc.r as master_rmag, cc.darcsec,p.objid,p.r as rmag,
               cc.ra,cc.dec,p.rowcErr * %f as raerr,p.colcErr * %f as decerr,F.mjd_r - %f as time,p.run,p.rerun,p.camcol,p.field
               from (SELECT p.r, rr0.colcErr * %f as master_decerr, rr0.rowcErr * %f as master_raerr, dbo.fdistancearcmineq(rr0.ra,rr0.dec,q.ra,q.dec)*60 as darcsec,q.ra,q.dec,
        p.htmid, q.objid,rr0.objid as rr0objid, rr0.ra as rr0ra,rr0.dec as qdec from
        (SELECT p.objid,p.ra,p.dec,p.rowc,p.rowcErr,p.colc,p.colcErr,
                p.u,p.g,p.r,p.i,p.z,
                p.run,p.rerun,p.camcol,p.field
                FROM dbo.fGetObjFromRect(%f,%f,%f,%f) n
                join PhotoObjAll p on n.objID=p.objID
                WHERE 
                n.run in (106,206)
                %s) as rr0, PhotoObj p,PhotoObj q 
                where rr0.objid = p.objid
                and q.htmid between 3000*(p.htmid/3000) and 3000*(p.htmid/3000+2) and
                q.objid != p.objid
                and dbo.fdistancearcmineq(rr0.ra,rr0.dec,q.ra,q.dec)*60 < 2 )  as cc
                join PhotoObjAll p on cc.objid=p.objID
                join Field F on F.fieldID = p.fieldID
                 -- order by
                 -- time
                
        """ % (sdss_coadd_platescale, sdss_coadd_platescale, t0, \
               sdss_coadd_platescale, sdss_coadd_platescale, ramin,ramax,decmin,decmax, rc)
        
        #print q
        rez = self.recquery(q)
        if savefile and not os.path.exists(master_name):
            np.savez(master_name,master=rez)
        return rez
        
    def test1(self):
        q = """SELECT TOP 10 objid,ra,dec,u,g,r,i,z,
            run,rerun,camcol,field
        FROM PhotoObj
        WHERE 
            u BETWEEN 0 AND 19.6 
            AND g BETWEEN 0 AND 20
            and run in (106,206)
        """
        print self._query(q).readlines()
        
    def test2(self):
        q = """SELECT TOP 10 objid,ra,dec,u,g,r,i,z,
            run,rerun,camcol,field
        FROM PhotoObj
        WHERE 
            u BETWEEN 0 AND 19.6 
            AND g BETWEEN 0 AND 20
            and run in (106,206)
        """
        return self.recquery(q)
            
    def test3(self):
        m = self.get_master_cat()
        t = self.write_match_files(m)
        
"""
In [4]: b[5].ra, b[5].dec
Out[4]: (342.19783344000001, -0.89816640000000003)

In [5]: b[0].ra, b[0].dec
Out[5]: (342.19151775, -0.90201332000000001)

/usr/stsci/wcstools-3.7.3/bin.macintel/skycoor -vr 342.19783344000001 -0.89816640000000003 342.19151775 -0.90201332000000001
Distance is 26.620 arcsec
dRA = -22.73368 arcsec, dDec = -13.84891 arcsec

In [8]: ((b[5].colc - b[0].colc)**2 + (b[5].rowc - b[0].rowc)**2)**(0.5)
Out[8]: 67.227760180758793

In [9]: 26.620/67.2277601807
Out[9]: 0.3959673790774629
"""