import os
from collections import defaultdict

import arcpy

import common

def otherParts(onePart):
  partDir = os.path.dirname(onePart)
  onePartName = os.path.basename(onePart)
  root = onePartName[:onePartName.find('.part')]
  partNames = [dir for dir in os.listdir(partDir) if dir.startswith(root + '.part')]
  return [os.path.join(partDir, partName) for partName in partNames]

def calcMergeDict(workspaces):
  mergeDict = defaultdict(list)
  for wsp in workspaces:
    for fc in common.listLayers(wsp):
      mergeDict[os.path.splitext(fc)[0]].append(common.featurePath(wsp, fc))
  return mergeDict
  
def mergeWorkspaces(partWsp, tgtWsp):
  common.progress('finding merge relationships')
  allSources = otherParts(partWsp)
  mergeDict = calcMergeDict(allSources)
  for name, sources in mergeDict.iteritems():
    tgt = common.featurePath(tgtWsp, name)
    if len(sources) == 1:
      common.progress('copying ' + name)
      arcpy.CopyFeatures_management(sources[0], tgt)
    else:
      common.progress('merging ' + name)
      arcpy.Merge_management(sources, tgt)

with common.runtool(2) as parameters:
  mergeWorkspaces(*parameters)