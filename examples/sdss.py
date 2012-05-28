import os, sys, string,urllib2,urllib,copy
from math import log10, radians, pi, sin, cos, atan2, sqrt
import time, traceback, datetime
import threading
import StringIO
import numpy as np
from matplotlib.mlab import csv2rec

class sdssq(object):

    #dr_url="http://cas.sdss.org/astrodr7/en/tools/search/x_sql.asp"
    dr_url="http://cas.sdss.org/stripe82/en/tools/search/x_sql.asp"
    formats = ['csv','xml','html']
    def_fmt = "csv"
    sdss_coadd_platescale = 0.396127 ## arcsec/pix
    

        
    
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
        print rez.readlines()
        rez.seek(0)
        tmp = rez.readline()
        rez.seek(0)
        if len(tmp) == 0 or tmp.find("error_message") != -1 or tmp.find("ERROR") != -1:
            print "rez:"
            print rez.readlines()
            return np.zeros((1,)).view(np.recarray)
        
        return csv2rec(rez)
        
    def _query(self,sql,url=dr_url,fmt=def_fmt):
        "Run query and return file object"

        fsql = self._filtercomment(sql)
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
        runids=[4198]):
        
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
        for x in master:
            d = self.dist(pos[0],pos[1],x["ra"],x["dec"])
            if d < mind:
                mind = d
                bestid = x["master_objid"]
            if mind < 2.5/(3600.0):
                gotit= True
                break
        
        if gotit:
            print "source of interest: mind=", mind, "bestid=", bestid
        else:
            print "couldn't find the source after %i sources searched" % len(master)
            
        #for r in runids:
        #    ## get only the matches
        #    tmp = master[np.where(master.run == r)]
        
        
    # 342.1913750  -0.9019444 J2000
    def get_master_cat(self,pos=(342.1913750, -0.9019444),errdeg=0.1,rcut=23.5,t0=5.21972623E4,\
        MASTER_PRE = "master_sdss",savefile=True):
        """
        issue the ugly query to get master catalog positions in stripe82 over time
        """
        master_name = MASTER_PRE + "%.4f_%.5f.npz" % pos

        if os.path.exists(master_name):
            npzfile = np.load(master_name)
            return npzfile["master"].view(np.recarray)
        
        ramin = pos[0] - 0.5*errdeg/np.cos(pos[1]) ; ramax = pos[0] + 0.5*errdeg/np.cos(pos[1]) 
        decmin = pos[1] - 0.5*errdeg ; decmax = pos[1] + 0.5*errdeg
        rc = "and p.r < %f" % rcut if rcut not in [None] else ""
        
        q = """SELECT cc.rr0objid as master_objid,cc.rr0ra as master_ra,cc.qdec as master_dec,
               cc.master_decerr,cc.master_raerr,cc.r as master_rmag, cc.darcsec,p.objid,p.r as rmag,
               cc.ra,cc.dec,p.rowcErr * 0.396127 as raerr,p.colcErr * 0.396127 as decerr,F.mjd_r - %f as time,p.run,p.rerun,p.camcol,p.field
               from (SELECT p.r, rr0.colcErr * 0.396127 as master_decerr, rr0.rowcErr * 0.396127 as master_raerr, dbo.fdistancearcmineq(rr0.ra,rr0.dec,q.ra,q.dec)*60 as darcsec,q.ra,q.dec,
        p.htmid, q.objid,rr0.objid as rr0objid, rr0.ra as rr0ra,rr0.dec as qdec from
        (SELECT p.objid,p.ra,p.dec,p.rowc,p.rowcErr,p.colc,p.colcErr,
                p.u,p.g,p.r,p.i,p.z,
                p.run,p.rerun,p.camcol,p.field
                FROM dbo.fGetObjFromRect(%f,%f,%f,%f) n
                join PhotoObjAll p on n.objID=p.objID
                WHERE 
                n.run in (106,206)
                %s) as rr0, star p,star q 
                where rr0.objid = p.objid
                and q.htmid between 2000*(p.htmid/2000) and 2000*(p.htmid/2000+1) and
                q.objid != p.objid
                and dbo.fdistancearcmineq(rr0.ra,rr0.dec,q.ra,q.dec)*60 < 2 )  as cc
                join PhotoObjAll p on cc.objid=p.objID
                join Field F on F.fieldID = p.fieldID
                 order by
                time
                
        """ % (t0,ramin,ramax,decmin,decmax, rc)
        
        print q
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