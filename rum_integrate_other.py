import common
import rum_integrate

with common.runtool(6) as parameters:
  uaLayer, popLayer, demTiles, clipping, workspace, prj = parameters
  demTiles = common.split(demTiles) # dem tile list
  rum_integrate.integrate(uaLayer, popLayer, demTiles, clipping, workspace, prj=prj, osm=False)