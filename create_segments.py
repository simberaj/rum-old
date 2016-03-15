# intersect barriers, demography, landuse
# select those with landuse code 11* or 121* or 133*
# filter out too small ones
# subdivide too spread out/large
# output

import arcpy
import common
import subdivide_polygons

def createSegments(landuse, otherFCs, codeFld, minsize, maxsize, maxwagner, output):
  with common.PathManager(output) as pathman:
    minsize, maxsize, maxwagner = (s.replace(',', '.') for s in (minsize, maxsize, maxwagner))
    # intersect everything
    common.progress('intersecting borders and barriers')
    intersected = pathman.tmpFile()
    arcpy.FeatureToPolygon_management([landuse] + otherFCs, intersected, attributes='NO_ATTRIBUTES')
    common.progress('clearing segment attributes')    
    common.clearFields(intersected)
    # select only suitable landuse
    common.progress('selecting urban landuse classes')
    inhabLU = pathman.tmpFile()
    inhabQry = common.safeQuery("[{0}] LIKE '11%' OR [{0}] LIKE '121%' OR [{0}] LIKE '133%'".format(codeFld), landuse)
    arcpy.Select_analysis(landuse, inhabLU, inhabQry)
    # join landuse information
    common.progress('joining landuse information')
    withLU = pathman.tmpFile()
    arcpy.SpatialJoin_analysis(intersected, inhabLU, withLU, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', '', 'WITHIN')
    common.progress('clearing segment attributes')    
    common.clearFields(withLU, [codeFld])
    # subdivide large polygons
    common.progress('subdividing large polygons')
    subdiv = pathman.tmpFile()
    subdivide_polygons.subdivide(withLU, subdiv, maxsize, maxwagner, None, [codeFld])
    # filter out small polygons
    common.progress('filtering out small polygons')
    common.ensureShapeAreaField(subdiv)
    arcpy.Select_analysis(subdiv, pathman.getOutputPath(), common.SHAPE_AREA_FLD + ' > ' + str(minsize))
    

with common.runtool(7) as parameters:
  landuse, otherFCsStr, codeFld, minsize, maxsize, maxwagner, output = parameters
  otherFCs = common.split(otherFCsStr)
  createSegments(landuse, otherFCs, codeFld, minsize, maxsize, maxwagner, output)

