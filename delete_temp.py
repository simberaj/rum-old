import common
import os

with common.runtool(1) as params:
  workspace = params.pop()
  common.debug(workspace)
  fcs = common.listLayers(workspace)
  todelete = [os.path.join(workspace, fc) for fc in fcs if fc.startswith('tmp_')]
  for fc in todelete:
    arcpy.Delete_management(fc)
