
============
Parsing data
============

pyBA works in catalogue space, with one catalogue of objects per image. Catalogues will often have been generated using SExtractor [link], and include the locations with uncertainity for each object as well as meta-information about the image. 

Catalogue I/O
=============

I/O commands are provided by the pyBA.parse module. 

parse.read()
  Reads columns

parse.write()
  Writes columns

*Interfacing with wcslib and PyFITS*
====================================

*A scheme for working with wcslib and PyFITS is not implemented yet.*
