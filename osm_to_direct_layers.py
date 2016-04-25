from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

import json
import os
import sys

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
  transformer = transform.GeoJSONTransformer(config, folder)
  common.progress('transforming OSM to layers')
  if osmfile.endswith('osm'):
    with transformer:
      osmListener = geojson.Bridge(transformer)
      osm.extract(osmfile, osmListener)
  elif osmfile.endswith('json'):
    with transformer:
      geojson.feed(osmfile, transformer)
    
def convertParts(motherfile, folder, config, prj=None, clipping=None):
  dirPath = os.path.abspath(os.path.dirname(motherfile))
  mothername = os.path.splitext(os.path.basename(motherfile))[0]
  pattern = mothername + '.part'
  parts = [file for file in os.listdir(dirPath) if file.startswith(pattern)]
  partDirs = [os.path.join(folder, os.path.splitext(part)[0]) for part in parts]
  for i in range(len(parts)):
    if not os.path.isdir(partDirs[i]):
      os.mkdir(partDirs[i])
    convert(os.path.join(dirPath, parts[i]), partDirs[i], config, prj, clipping)

if __name__ == '__main__':
  import common
  with common.runtool(6) as parameters:
    if common.toBool(parameters[5], 'use part files switch'):
      convertParts(*parameters[:5])
    else:
      import cProfile
      oldStdout = sys.stdout
      sys.stdout = open('t:\\translog.log', 'w')
      cProfile.run('convert(*parameters[:5])')
      sys.stdout = oldStdout
      # convert(*parameters[:5])
  # parser = argparse.ArgumentParser(description=DESCRIPTION)
  # parser.add_argument('osmFile', metavar='osm', type=unicode, help='an OSM file whose contents are to be translated')
  # parser.add_argument('outFolder', metavar='workspace', type=unicode, help='an ArcGIS workspace to place output and intermediary files')
  # parser.add_argument('config', metavar='config', help='a JSON file specifying what should be translated to what')
  # # parser.add_argument('-c', '--crs', dest='crs', default='4326', help='EPSG code of output data projection, defaults to 4326 (WGS-84 latlon)')
  # # parser.add_argument('-e', '--enc', dest='encoding', default='utf8', help='output encoding (in python format), default is UTF-8 (utf8)')
  # args = parser.parse_args()
  # convert(args.osmFile, args.outFolder, args.config, args.prj)