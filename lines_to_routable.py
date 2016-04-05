import arcpy
import common
import loaders

def linesToRoutable(lines, outfile, levelFld=None, groundLevel=None, transferFlds=[], builtup=None):
  with common.PathManager(outfile) as pathman:
    dissolved = pathman.tmpFile()
    dissolveFlds = transferFlds + ([levelFld] if levelFld else [])
    common.message(transferFlds)
    common.progress('homogenizing lines')
    arcpy.Dissolve_management(lines, dissolved, dissolveFlds, [], 'SINGLE_PART', 'UNSPLIT_LINES')
    if levelFld:
      connectLinesByLevels(dissolved, levelFld, outfile, groundLevel, shout=True)
    else:
      common.progress('connecting lines')
      arcpy.FeatureToLine_management(dissolved, outfile)
    common.progress('cleaning output attributes')
    common.clearFields(outfile, exclude=dissolveFlds)

def connectLinesByLevels(lines, levelFld, outfile, groundLevel=None, shout=False):
  if shout: common.progress('loading levels')
  levels = loaders.getUniqueValues(lines, levelFld)
  with common.PathManager(outfile) as pathman:
    # create feature class for each level
    if shout: common.progress('separating levels')
    levelFCs = []
    groundLevelI = None
    for level in levels:
      levelFCs.append(pathman.tmpFile())
      arcpy.Select_analysis(lines, levelFCs[-1], common.safeQuery("[{}]={}".format(levelFld, "'" + level + "'" if isinstance(level, str) else level), lines))
      if groundLevel is not None and str(level) == str(groundLevel):
        groundLevelI = len(levelFCs) - 1
    # cut lines at intersections, only on ground level
    if groundLevelI is not None:
      if shout: common.progress('connecting ground lines')
      connectedGroundFC = pathman.tmpFile()
      arcpy.FeatureToLine_management(levelFCs[groundLevelI], connectedGroundFC)
      levelFCs[groundLevelI] = connectedGroundFC
    # import os
    # arcpy.Merge_management(connected, os.path.join(os.path.dirname(outfile), 'connected.shp'))
    # generate endpoints to cut the other levels
    if shout: common.progress('generating level endpoints')
    endpointFCs = []
    for levelFC in levelFCs:
      endpointFCs.append(pathman.tmpFile())
      arcpy.FeatureVerticesToPoints_management(levelFC, endpointFCs[-1], 'BOTH_ENDS')
    # arcpy.Merge_management(endpointFCs, os.path.join(os.path.dirname(outfile), 'cutpoints.shp'))
    # merge the endpoints
    endpoints = pathman.tmpFile()
    arcpy.Merge_management(endpointFCs, endpoints)
    # cut the other levels
    if shout: common.progress('connecting levels')
    cutFCs = []
    for connFC in levelFCs:
      cutFCs.append(pathman.tmpFile())
      arcpy.SplitLineAtPoint_management(connFC, endpoints, cutFCs[-1], '1 Decimeters')
    # merge to final result
    if shout: common.progress('merging levels')
    arcpy.Merge_management(cutFCs, outfile)

    
if __name__ == '__main__':
  with common.runtool(4) as parameters:
    lines, levelFld, transferFlds, outfile = parameters
    linesToRoutable(lines, outfile, levelFld, common.parseFields(transferFlds))
    
