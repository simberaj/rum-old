import common
import arcpy
import networking
import loaders
import collections
import stats

MAX_LOCATIONS = 1000
TOLERANCE = '1500 Meters'

def createSubdividedCentroids(inputs, target, count=20):
  import geometry
  inputPts = loaders.OneFieldReader(inputs, loaders.SHAPE_SLOT).read()
  poly = geometry.MultiPolygon.asConvexHull(inputPts)
  centroids = [chunk.centroidAsList() for chunk in poly.subdivide(maxarea=(poly.area / float(count)))]
  # print(centroids)
  loaders.OneFieldWriter(target, loaders.SHAPE_SLOT, crs=inputs).write(centroids)
  
def accessibility(origins, network, impedance, testTargets=None, testCount=20, accFld='ACC', shout=True):
  arcpy.CheckOutExtension('Network')
  with common.PathManager(origins) as pathman:
    # create target points deterministically: centroids of subdivided convex hull
    if not testTargets:
      if shout: common.progress('creating accessibility testing targets')
      testTargets = pathman.tmpFile()
      createSubdividedCentroids(origins, testTargets, count=testCount)
    origIDFld = pathman.tmpField(origins, int)
    common.copyField(origins, common.ensureIDField(origins), origIDFld)
    tgtIDFld = common.ensureIDField(testTargets)
    # calculate origin network locations
    sources = networking.Networker.getNetworkSources(network)
    common.debug(sources, TOLERANCE, common.count(origins))
    if shout: common.progress('snapping places to network')
    arcpy.CalculateLocations_na(origins, network, TOLERANCE, sources)
    if shout: common.progress('snapping testing targets to network')
    arcpy.CalculateLocations_na(testTargets, network, TOLERANCE, sources)
    # split the origins into multiple if too many
    origCount = common.count(origins)
    if origCount > MAX_LOCATIONS:
      splitCount = int(origCount / MAX_LOCATIONS) + 1
      if shout: common.progress('splitting place dataset to {} parts'.format(splitCount))
      ids = loaders.OneFieldReader(origins, origIDFld).read()
      ids.sort()
      splits = [ids[0] - 1] + [ids[j * len(ids) / splitCount - 1] for j in range(1, splitCount+1)]
      common.debug(splits)
      originList = []
      if shout: prog = common.progressor('splitting by ID', splitCount)
      for i in range(len(splits)-1):
        originList.append(pathman.tmpFile())
        arcpy.Select_analysis(origins, originList[-1], '{0} > {1} AND {0} <= {2}'.format(origIDFld, splits[i], splits[i+1]))
        if shout: prog.move()
      if shout: prog.end()
    else:
      originList = [origins]
    # start calculating the analysis
    if shout: common.progress('creating analysis layer')
    naLayer = arcpy.MakeODCostMatrixLayer_na(network, pathman.tmpLayer(), impedance, accumulate_attribute_name=impedance, output_path_shape='NO_LINES').getOutput(0)
    # naLayer = arcpy.MakeODCostMatrixLayer_na(network, pathman.tmpLayer(), impedance, accumulate_attribute_name=impedance, output_path_shape='NO_LINES').getOutput(0)
    results = common.sublayer(naLayer, 'Lines')
    # print(arcpy.na.GetNAClassNames(ly))
    resultSlots = {'ids' : 'Name', 'cost' : 'Total_' + impedance}
    origMap = common.NET_FIELD_MAPPINGS + ';Name {} #'.format(origIDFld)
    tgtMap = common.NET_FIELD_MAPPINGS + ';Name {} #'.format(tgtIDFld)
    common.debug(origMap)
    common.debug(tgtMap)
    costs = collections.defaultdict(list)
    medianCosts = {}
    # kajdfkla
    if shout: prog = common.progressor('calculating accessibility', len(originList))
    for originPart in originList:
      arcpy.AddLocations_na(naLayer, 'Origins', originPart, origMap, '', append='CLEAR')
      arcpy.AddLocations_na(naLayer, 'Destinations', testTargets, tgtMap, '', append='CLEAR')
      arcpy.Solve_na(naLayer)
      # print(naLayer, results)
      # print(common.fieldList(results))
      data = loaders.BasicReader(results, resultSlots).read()
      common.debug(data)
      for item in data:
        origID = int(item['ids'].split(' - ')[0])
        costs[origID].append(item['cost'])
      common.debug(costs)
      for origID, costList in costs.iteritems():
        medianCosts[origID] = stats.median(costList)
      costs = collections.defaultdict(list)
      if shout: prog.move()
    if shout: prog.end()
    common.debug(medianCosts)
    if shout: common.progress('writing accessibility values')
    # common.debug(medianCosts)
    meanCost = float(sum(medianCosts.itervalues())) / len(medianCosts)
    access = collections.defaultdict(lambda: {'acc' : 0.0},
      {key : {'acc' : meanCost / cost} for key, cost in medianCosts.iteritems()})
    loaders.ObjectMarker(origins, {'id' : origIDFld}, {'acc' : accFld}).mark(access)

    
if __name__ == '__main__':
  with common.runtool(4) as parameters:
    accessibility(*parameters)
  # with common.runtool(3) as parameters:
    # createSubdividedCentroids(*parameters)