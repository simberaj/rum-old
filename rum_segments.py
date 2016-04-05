import common
import create_segments

def segments(workspace, minsize, maxsize, maxwagner):
  landuse = common.featurePath(workspace, 'ua')
  otherFCs = [common.featurePath(workspace, 'barrier')]
  uaCodeFlds = [fld for fld in common.fieldList(landuse) if fld.lower().startswith('code') and fld.lower() != 'code_int']
  if not uaCodeFlds:
    raise ValueError, 'could not detect land use code field in Urban Atlas layer'
  uaCodeFld = uaCodeFlds.pop()
  create_segments.createSegments(landuse, otherFCs, uaCodeFld, minsize, maxsize, maxwagner, common.featurePath(workspace, 'segments'))
  
if __name__ == '__main__':
  with common.runtool(4) as parameters:
    workspace, minsize, maxsize, maxwagner = parameters
    segments(workspace, common.toFloat(minsize, 'minimum segment size'), common.toFloat(maxsize, 'maximum segment size'), common.toFloat(maxwagner, 'maximum segment Wagner index'))