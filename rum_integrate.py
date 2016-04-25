import common
import osm
import osm_to_direct_layers
import os
import loaders
import arcpy

def integrate(uaLayer, popLayer, demTiles, clipping, workspace, transformConf=None, prj=None, osm=True):
  if not prj:
    common.warning('No explicit coordinate system specified, using Urban Atlas CRS: ETRS-89/LAEA')
    prj = arcpy.Describe(uaLayer).spatialReference
  else:
    newprj = arcpy.SpatialReference(4326) # we need to create a dummy instance
    newprj.loadFromString(prj)
    prj = newprj
    # common.debug(prj)
    # prj = arcpy.Describe(uaLayer).spatialReference.loadFromString(prj)
    # common.debug(prj)
    # common.debug(prj.name)
  if not clipping:
    common.warning('No explicit extent specified, using Urban Atlas layer extent')
    clipping = common.featurePath(workspace, 'extent')
    createClipping(uaLayer, clipping, prj)
  elif prj.factoryCode != arcpy.Describe(clipping).spatialReference.factoryCode:
    # todo match by EPSG code
    common.progress('projecting analysis extent')
    arcpy.Project_management(clipping, common.featurePath(workspace, 'extent'), prj)
  if osm:
    if not transformConf:
      raise ValueError, 'OSM download requested but no transformation configuration supplied'
    integrateOSM(workspace, transformConf, clipping, prj)
  integrateUA(workspace, uaLayer, clipping, prj)
  if popLayer:
    integratePop(workspace, popLayer, clipping, prj)
  integrateDEM(workspace, demTiles, clipping, prj)
  
def integrateOSM(workspace, transformConf, clipping, prj):
  common.progress('downloading OSM data')
  targetOSM = osmFilePath(common.folder(workspace))
  tiles = list(loaders.findExtentTiles(clipping, osm.MAX_TILE_SIZE_DEG, osm.MAX_TILE_SIZE_DEG))
  common.message('Found {} tiles to download.'.format(len(tiles)))
  osm.downloadTiles(tiles, targetOSM, prog=common.progressor('downloading tiles', len(tiles)))
  common.progress('converting OSM data')
  osm_to_direct_layers.convert(targetOSM, workspace, transformConf, prj=prj, clipping=clipping)
  # return transformConf['layers'].keys()

def integrateUA(workspace, uaLayer, clipping, prj):
  uaCRS = arcpy.Describe(uaLayer).spatialReference
  outPath = common.featurePath(workspace, 'ua')
  if clipping != uaLayer or prj != uaCRS:
    with common.PathManager(clipping) as pathman:
      if clipping != uaLayer:
        clipped = pathman.tmpFC()
        common.progress('clipping Urban Atlas layer')
        arcpy.Clip_analysis(uaLayer, clipping, clipped)
      else:
        clipped = uaLayer
      if prj != uaCRS:
        common.progress('projecting Urban Atlas layer')
        arcpy.Project_management(clipped, outPath, prj)
      else:
        common.progress('copying Urban Atlas layer')
        arcpy.CopyFeatures_management(clipped, outPath)
  else:
    common.progress('copying Urban Atlas layer')
    arcpy.CopyFeatures_management(uaLayer, outPath)
  return ['ua']
  
def integratePop(workspace, popLayer, clipping, prj):
  common.progress('clipping population layer')
  popSel = common.PathManager(clipping).tmpLayer()
  arcpy.MakeFeatureLayer_management(popLayer, popSel)
  arcpy.SelectLayerByLocation_management(popSel, 'INTERSECT', clipping)
  outPath = common.featurePath(workspace, 'pop')
  if prj != arcpy.Describe(popSel).spatialReference:
    arcpy.Project_management(popSel, outPath, prj)
  else:
    arcpy.CopyFeatures_management(popSel, outPath)
  return ['pop']

def integrateDEM(workspace, demTiles, clipping, prj):
  outPath = common.rasterPath(workspace, 'dem')
  extent = arcpy.Describe(clipping).extent
  with common.PathManager(clipping) as pathman:
    if len(demTiles) > 1:
      tile = pathman.tmpRaster()
      common.progress('merging DEM tiles')
      arcpy.Mosaic_management(demTiles, tile)
    else:
      tile = demTiles[0]
    tileCRS = arcpy.Describe(tile).spatialReference
    common.debug(prj)
    common.debug(tileCRS)
    if prj != tileCRS:
      common.progress('projecting clipping area')
      clipInDem = pathman.tmpFC()
      arcpy.Project_management(clipping, clipInDem, tileCRS)
      common.progress('clipping DEM')
      clippedRas = pathman.tmpRaster()
      arcpy.Clip_management(tile, '', clippedRas, clipInDem)
      common.progress('projecting DEM')
      arcpy.ProjectRaster_management(clippedRas, outPath, prj, 'CUBIC')
    else:
      common.progress('clipping DEM')
      arcpy.Clip_management(tile, clipping, outPath)
  
def osmFilePath(folder):
  return os.path.join(folder, os.path.basename(folder) + '.osm')

def createClipping(layer, target, prj):
  common.progress('creating clipping extent')
  if prj != arcpy.Describe(layer).spatialReference:
    with common.PathManager(target) as pathman:
      diss = pathman.tmpFC()
      arcpy.Dissolve_management(layer, diss)
      arcpy.Project_management(diss, target, prj)
  else:
    arcpy.Dissolve_management(layer, target)
      
  
if __name__ == '__main__':
  with common.runtool(7) as parameters:
    parameters[2] = common.split(parameters[2]) # dem tile list
    integrate(*parameters)
    
