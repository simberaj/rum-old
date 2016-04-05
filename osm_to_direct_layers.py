from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

import json

import osm
import geojson
import transform
import common

DEFAULT_ENCODING = 'utf8'

DESCRIPTION = '''OSM direct layer translator. Converts an .osm file to ArcGIS layers based on the given configuration.'''

def convert(osmfile, folder, config, prj=None, clipping=None):
  import os, common, arcpy
  if prj or clipping or common.isInDatabase(folder):
    with common.PathManager(folder) as pathman:
      subfolder = pathman.tmpSubfolder()
      # print(subfolder)
      _convert(osmfile, subfolder, config)
      lyrlist = common.listLayers(subfolder)
      prog = common.progressor('adjusting spatial reference', len(lyrlist))
      for fc in lyrlist:
        fcPath = os.path.join(subfolder, fc)
        tgtPath = common.featurePath(folder, os.path.splitext(fc)[0])
        if prj or clipping:
          if prj and clipping:
            projToPath = pathman.tmpFile()
            arcpy.Project_management(fcPath, projToPath, prj)
            # print(projToPath, clipping, tgtPath)
            arcpy.Clip_analysis(projToPath, clipping, tgtPath)
          elif prj:
            arcpy.Project_management(fcPath, tgtPath, prj)
          else:
            arcpy.Clip_analysis(fcPath, clipping, tgtPath)
        else:
          arcpy.CopyFeatures_management(fcPath, tgtPath)
        prog.move()
      prog.end()
  else:
    _convert(osmfile, folder, config)
 
def _convert(osmfile, folder, config):
  common.progress('loading layer transform configuration')
  with open(config) as conffile:
    config = json.load(conffile, encoding=DEFAULT_ENCODING)
  transformer = transform.Transformer(config, folder)
  common.progress('transforming OSM to layers')
  if osmfile.endswith('osm'):
    osmListener = geojson.Bridge(transformer)
    osm.extract(osmfile, osmListener)
  elif osmfile.endswith('json'):
    geojson.feed(osmfile, transformer)
    
  

if __name__ == '__main__':
  import common
  with common.runtool(5) as parameters:
    # import cProfile
    # cProfile.run('convert(*parameters)')
    convert(*parameters)
  # parser = argparse.ArgumentParser(description=DESCRIPTION)
  # parser.add_argument('osmFile', metavar='osm', type=unicode, help='an OSM file whose contents are to be translated')
  # parser.add_argument('outFolder', metavar='workspace', type=unicode, help='an ArcGIS workspace to place output and intermediary files')
  # parser.add_argument('config', metavar='config', help='a JSON file specifying what should be translated to what')
  # # parser.add_argument('-c', '--crs', dest='crs', default='4326', help='EPSG code of output data projection, defaults to 4326 (WGS-84 latlon)')
  # # parser.add_argument('-e', '--enc', dest='encoding', default='utf8', help='output encoding (in python format), default is UTF-8 (utf8)')
  # args = parser.parse_args()
  # convert(args.osmFile, args.outFolder, args.config, args.prj)