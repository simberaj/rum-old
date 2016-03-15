import os, common, arcpy

TMP_DICED_NAME = 'tmp_diced'
TMP_BUFFERED_NAME = 'tmp_buffered'
TMP_MERGED_NAME = 'tmp_merged'
TMP_FILLED_NAME = 'tmp_filled'
TMP_DISSOLVED_NAME = 'tmp_dissolved'
MAX_VERTEX_COUNT = 10000
SIMPLIFY_FACTOR = 4.0

with common.runtool(4) as parameters:
  common.progress('loading parameters and setup')
  buildings, bufferDist, minSize, output = parameters
  with common.PathManager(output) as pathman:
    # tmpDiced = common.addFeatureExt(os.path.join(workspace, TMP_DICED_NAME))
    # tmpBuffered = common.addFeatureExt(os.path.join(workspace, TMP_BUFFERED_NAME))
    # # tmpMerged = common.addFeatureExt(os.path.join(workspace, TMP_MERGED_NAME))
    # tmpFilled = common.addFeatureExt(os.path.join(workspace, TMP_FILLED_NAME))
    # tmpDissolved = common.addFeatureExt(os.path.join(workspace, TMP_DISSOLVED_NAME))
    common.progress('subdividing large buildings and areas')
    # common.debug(buildings)
    # common.debug(tmpDiced)
    # common.debug(MAX_VERTEX_COUNT)
    tmpDiced = pathman.tmpFile()
    arcpy.Dice_management(buildings, tmpDiced, MAX_VERTEX_COUNT)
    common.progress('indexing features')
    arcpy.AddSpatialIndex_management(tmpDiced)
    common.progress('buffering buildings and areas')
    oldtol = arcpy.env.XYTolerance
    auxtol = bool(oldtol is None)
    if auxtol:
      oldtol = bufferDist
    arcpy.env.XYTolerance = str(float(bufferDist.split()[0]) / 10) + ' ' + oldtol.split()[1]
    tmpBuffered = pathman.tmpFile()
    arcpy.Buffer_analysis(buildings, tmpBuffered, bufferDist, '', '', 'ALL')
    # arcpy.Buffer_analysis(buildings, tmpBuffered, bufferDist, '', '', 'NONE')
    common.progress('indexing features')
    arcpy.AddSpatialIndex_management(tmpBuffered)
    # common.progress('merging buffers')
    # arcpy.Dissolve_management(tmpBuffered, tmpMerged, '', '', 'SINGLE_PART')
    # arcpy.env.XYTolerance = (None if auxtol else oldtol)
    # common.progress('indexing features')
    # arcpy.AddSpatialIndex_management(tmpMerged)
    common.progress('filling holes')
    tmpFilled = pathman.tmpFile()
    # arcpy.Union_analysis([tmpMerged], tmpFilled, 'ONLY_FID', '', 'NO_GAPS')
    arcpy.Union_analysis([tmpBuffered], tmpFilled, 'ONLY_FID', '', 'NO_GAPS')
    common.progress('indexing features')
    arcpy.AddSpatialIndex_management(tmpFilled)
    common.progress('detecting small polygons')
    tmpDissolved = pathman.tmpFile()
    arcpy.Dissolve_management(tmpFilled, tmpDissolved, '', '', 'SINGLE_PART')
    common.progress('simplifying area borders')
    arcpy.SimplifyPolygon_cartography(tmpDissolved, output, 'POINT_REMOVE', common.multiplyDistance(bufferDist, 1 / SIMPLIFY_FACTOR), minSize, 'RESOLVE_ERRORS', 'NO_KEEP')
  # common.progress('deleting temporary files')
  # common.delete(tmpBuffered, tmpMerged, tmpDissolved, tmpFilled, tmpDiced)