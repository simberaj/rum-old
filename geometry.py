from __future__ import print_function, unicode_literals
import sys
import math
import numpy
import random
import operator

COOR_EPSILON = 1e-12

class Triangle:
  def __init__(self, pts):
    self.pts = pts
  
  @property
  def centroid(self):
    return sum(self.pts) / 3.0
  
  @property
  def area(self):
    return 0.5 * abs(numpy.cross(self.pts[2] - self.pts[0], self.pts[1] - self.pts[0]))
  
  def __repr__(self):
    return 'Triangle' + str(tuple(list(pt) for pt in self.pts))

class MultiPolygon:
  MAX_SPLIT_ITER = 50

  def __init__(self, coors):
    self.coors = coors
    self._area = None
    self._cf = None
    self._centroid = None
  
  def explode(self):
    return [MultiPolygon([part]) for part in self.coors]
  
  def isMultipart(self):
    return (len(self.coors) > 1)
  
  def getPart(self, i):
    return MultiPolygon(self.coors[i])
  
  def hasHoles(self, i=None):
    if i is None:
      for part in self.coors:
        if len(part) > 1:
          return True
      return False
    else:
      return (len(self.coors[i]) > 1)
  
  def cutHoles(self):
    for partI in xrange(len(self.coors)):
      outer = self.coors[partI][0]
      for hole in self.coors[partI][1:]:
        minPair = None
        minDist = sys.float_info.max
        for i in xrange(len(hole) - 1):
          for j in xrange(len(outer) - 1):
            dist = numpy.linalg.norm(hole[i] - outer[j])
            if dist < minDist:
              minPair = (i, j)
              minDist = dist
        if minPair:
          i, j = minPair
          outer = outer[:(j+1)] + hole[i:] + hole[1:(i+1)] + outer[j:]
      self.coors[partI] = [outer]
    return self
    
 
  def densify(self, maxLen=sys.float_info.max):
    for part in self.coors:
      for ring in part:
        i = 0
        while i < len(ring) - 1:
          dirvec = ring[i+1] - ring[i]
          edgeLen = numpy.linalg.norm(dirvec)
          if edgeLen > maxLen:
            partCount = int(math.floor(edgeLen / float(maxLen)))
            start = ring[i]
            ring[(i+1):(i+1)] = [start + k * dirvec / float(partCount) for k in xrange(1, partCount)]
            i += partCount
          else:
            i += 1
  
  def sparsify(self):
    '''Removes all points of the polygon that do not change the polygon shape
    (lie on a straight line connecting their predecessor and successor).'''
    for part in self.coors:
      for ring in part:
        i = 1
        while i < len(ring) - 1:
          if onLine(ring[i], ring[i-1], ring[i+1]) and len(ring) > 4:
            ring.pop(i)
          else:
            i += 1
  
  @staticmethod
  def generalize(ring, minLen=0):
    ring = ring[:]
    i = 0
    while i < len(ring) - 1:
      dirvec = ring[i+1] - ring[i]
      edgeLen = numpy.linalg.norm(dirvec)
      if edgeLen < minLen:
        ring.pop(i+1)
      else:
        i += 1
    return ring
  
  def subdivide(self, minarea=1, maxarea=sys.float_info.max, maxwagner=sys.float_info.max):
    return Subdivider(minarea, maxarea, maxwagner).subdivide(self)
  
  def centroidAsList(self):
    return list(self.centroid)
  
  @property
  def centroid(self):
    if self._centroid is None:
      self._centroid = self._calcCentroid()
    return self._centroid
  
  def _calcCentroid(self):
    self.sparsify()
    # print('sparsified')
    # print(self.coors)
    # print(convexHull(self.coors[0][0]))
    centroid = numpy.zeros(2)
    trianglesArea = 0.0
    for tr in self.triangulateSimple():
      # print(tr)
      centroid += tr.centroid * (tr.area / self.area) # dividing by self.area to avoid overflow
      trianglesArea += tr.area
    return centroid * self.area / trianglesArea
  
  def triangulateSimple(self):
    # traingulates only the outer ring, ignoring holes (used only for centroid calculation at the moment)
    for part in self.coors:
      try:
        for triangle in convexTriangulation(convexHull(part[0][:-1])):
          yield triangle
      except ValueError:
        print(part[0])
        print(self.coors)
        raise
          
  @property
  def circumference(self):
    if self._cf is None:
      self._cf = self._calcCircumference()
    return self._cf

  def _calcCircumference(self):
    return sum(self.simpleCircumference(part[0]) + 
        sum(self.simpleCircumference(hole) for hole in part[1:])
      for part in self.coors)
      
  @property
  def area(self):
    if self._area is None:
      self._area = self._calcArea()
    return self._area
  
  def _calcArea(self):
    return sum(self.simpleArea(part[0]) - sum(self.simpleArea(hole) for hole in part[1:]) for part in self.coors)
  
  def setArea(self, area):
    self._area = area
    
  @staticmethod
  def simpleArea(p):
    '''Calculates the area of a non-self-intersecting singlepart polygon without holes.'''
    rgarr = numpy.array(p)
    # print(rgarr)
    return 0.5 * abs((rgarr[:-1,0] * rgarr[1:,1] - rgarr[:-1,1] * rgarr[1:,0]).sum())
    # return 0.5 * abs(sum(numpy.cross(p[i], p[i+1]) for i in xrange(len(p) - 1)))

  @staticmethod
  def simpleCircumference(p):
    '''Calculates the circumference of a non-self-intersecting singlepart polygon without holes.'''
    return sum(numpy.linalg.norm(p[i+1] - p[i]) for i in xrange(len(p) - 1))
    
  @property
  def wagner(self):
    return self.circumference / (2 * math.sqrt(math.pi * self.area))
  
  @classmethod
  def fromList(cls, coorsIn, safe=False):
    # print(coorsIn)
    coors = []
    for partIn in coorsIn:
      part = []
      for i in range(len(partIn)):
        part.append(fromList(partIn[i]))
        if not safe:
          setWinding(part[-1], clockwise=(i != 0))
      coors.append(part)
    return cls(coors)
  
  def toList(self):
    return [[[list(pt) for pt in ring] for ring in part] for part in self.coors]
  
  @classmethod
  def asConvexHull(cls, ptsIn):
    hull = convexHull(fromList(ptsIn))
    ring = hull + [hull[0]]
    setWinding(ring, False)
    return cls([[ring]])
  
  @property
  def bbox(self):
    xs = []
    ys = []
    for part in self.coors:
      for ring in part:
        for pt in ring:
          xs.append(pt[0])
          ys.append(pt[1])
    return (min(xs), min(ys), max(xs), max(ys))

    
def onLine(pt, edge1Pt, edge2Pt):
  return bool(abs(position(pt, edge1Pt, edge2Pt)) < COOR_EPSILON)
  
def position(pt, edge1Pt, edge2Pt): # returns 0 when on the line, >0 when left, <0 when right
  d1 = edge1Pt - pt
  d2 = edge2Pt - pt
  return d1[0] * d2[1] - d2[0] * d1[1]

  
def crosses(e1p1, e1p2, e2p1, e2p2):
  v1 = e1p2 - e1p1
  v2 = e2p2 - e2p1
  dircross = numpy.cross(v1, v2)
  if dircross == 0: # direction vectors linearly dependent (collinear/parallel)
    return False # only cross themselves if collinear and overlapping, no interest in that
  else:
    startDiff = e2p1 - e1p1
    t = numpy.cross(startDiff, v1) / dircross
    if t < 0 or t > 1:
      return False
    else:
      u = numpy.cross(startDiff, v2) / dircross
      if u < 0 or u > 1:
        return False
      else:
        return True

def vectorBetween(vec, vecFrom, vecTo):
  '''Returns True if the vec, taken from the same origin, is between
  vecFrom and vecTo, going counterclockwise.'''
  phi, phiFrom, phiTo = (math.atan2(v[1], v[0]) for v in (vec, vecFrom, vecTo))
  pi2 = 2 * math.pi
  phi += (pi2 if (phi < phiFrom or phi > (phiFrom + pi2)) else 0)
  phiTo += (pi2 if (phiTo < phiFrom or phiTo > (phiFrom + pi2)) else 0)
  return (phi < phiTo)
  
def pointInPolygon(pt, poly):
  inside = False
  x, y = pt
  p1x, p1y = poly[0]
  for i in xrange(len(poly)):
    p2x, p2y = poly[i % (len(poly)-1)]
    if y > min(p1y,p2y):
      if y <= max(p1y,p2y):
        if x <= max(p1x,p2x):
          if p1y != p2y:
            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
          if p1x == p2x or x <= xints:
            inside = not inside
    p1x,p1y = p2x,p2y
  return inside

def centroid(geojson):
  geomType = geojson['type']
  coors = geojson['coordinates']
  if geomType == 'LineString':
    return coors[int(len(coors) / 2)]
  elif geomType == 'MultiLineString':
    lenAll = sum(len(part) for part in coors)
    for part in coors:
      if lenAll < len(part):
        return part[lenAll]
      else:
        lenAll -= len(part)
  elif geomType == 'Polygon':
    return MultiPolygon.fromList([coors]).centroidAsList()
  elif geomType == 'MultiPolygon':
    return MultiPolygon.fromList(coors).centroidAsList()
  elif geomType == 'Point':
    return coors
  elif geomType == 'MultiPoint':
    return list(sum(fromList(coors)) / len(coors))
  else:
    raise ValueError, 'geometry type {} not recognized'.format(geomType)
    

def _adjust_dim(amin, amax, cellsize):
  firstToLast = int((amax - amin) / cellsize) * cellsize
  hang = ((amax - amin) - firstToLast) / 2
  return (amin + hang, amax - hang)
  
def grid(xmin, ymin, xmax, ymax, cellsize):
  xmin, xmax = _adjust_dim(xmin, xmax, cellsize)
  ymin, ymax = _adjust_dim(ymin, ymax, cellsize)
  x = xmin
  while x <= xmax:
    y = ymin
    while y <= ymax:
      yield (x, y)
      y += cellsize
    x += cellsize

def isClockwise(ring):
  return sum((ring[i+1][0] - ring[i][0]) * (ring[i+1][1] + ring[i][1]) for i in range(len(ring)-1)) > 0

def setWinding(ring, clockwise=True):
  if isClockwise(ring) != clockwise:
    ring.reverse()
  
def hasSelfIntersections(ring):
  # print('checking self intersection')
  ring = fromList(ring)
  for i in range(len(ring)-2):
    for j in range(i+2, len(ring)-1):
      if crosses(ring[i], ring[i+1], ring[j], ring[j+1]):
        # print('found self intersection')
        return True
  return False
        
def edgeWithinRing(ring, i, j):
  if i > j: i, j = j, i
  if vectorBetween(ring[j] - ring[i], ring[i+1] - ring[i], ring[i-1 if i != 0 else -2] - ring[i]):
    for k in range(i-1) + range(i+1, j-1) + range(j+1, len(ring)-1):
      if crosses(ring[i], ring[j], ring[k], ring[k+1]):
        # print(i, j, k, k+1)
        return False
    return True
  else:
    return False
    
def convexHullJarvis(points):
  '''Uses Jarvis-march algorithm (gift wrapping).'''
  print(points)
  i = 0
  import matplotlib.pyplot as plt
  plt.figure().gca().set_aspect(True)
  plt.ion()
  plt.show()
  plt.plot([pt[0] for pt in points], [pt[1] for pt in points], 'b')
  minx = min(pt[0] for pt in points)
  hull = [min([pt for pt in points if pt[0] == minx], key=operator.itemgetter(1))]
  plt.plot(hull[0][0], hull[0][1], 'g.')
  endpoint = points[0]
  while (hull[-1] != hull[0]).any() or len(hull) == 1:
    print(hull[0], hull[-1], len(hull), (hull[-1] - hull[0]))
    for j in range(1, len(points)):
      if (endpoint == hull[-1]).all() or position(points[j], hull[i], endpoint) > COOR_EPSILON:
        endpoint = points[j]
    i += 1
    plt.plot(endpoint[0], endpoint[1], 'r.')
    print(endpoint)
    x = raw_input()
    hull.append(endpoint)
  print('HULL', hull)
  return hull

# def angleWithX(pt1, pt2):
  # diff = pt2 - pt1
  # denom = (diff ** 2).sum()
  # return diff[0] / denom if denom else 2.0 # ensures pt1 goes first
  
def convexHull(points):
  '''Computes a convex hull of a set of points
     using the Andrew's monotone chain algorithm.
     Requires points as numpy arrays on input.'''
  # import matplotlib.pyplot as plt
  # plt.figure().gca().set_aspect(True)
  # plt.ion()
  # plt.show()
  # plt.plot([pt[0] for pt in points], [pt[1] for pt in points], 'b')
  points.sort(key=operator.itemgetter(1)) # sort by y
  points.sort(key=operator.itemgetter(0)) # then by x
  lower = []
  for i in range(len(points)):
    while len(lower) >= 2 and position(lower[-2], lower[-1], points[i]) <= 0:
      lower.pop()
    lower.append(points[i])
  upper = []
  for i in reversed(range(len(points))):
    while len(upper) >= 2 and position(upper[-2], upper[-1], points[i]) <= 0:
      upper.pop()
    upper.append(points[i])
  # upper.pop() - do not do this to have the same start and end
  lower.pop()
  # plt.plot([pt[0] for pt in lower+upper], [pt[1] for pt in lower+upper], 'r.')
  # plt.plot([pt[0] for pt in upper], [pt[1] for pt in upper], 'g')
  # x = raw_input()
  return lower + upper
    
  
def convexTriangulation(points):
  start = points[0]
  for i in xrange(1, len(points)-2):
    yield Triangle((start, points[i], points[i+1]))
  if (points[-1] != points[0]).any():
    yield Triangle((start, points[-2], points[-1]))

def subdivide(multipolygon, minarea=1,
    maxarea=sys.float_info.max, maxwagner=sys.float_info.max):
  return Subdivider(minarea, maxarea, maxwagner).subdivideList(multipolygon)
  
def fromList(points):
  return [numpy.array(pt) for pt in points]
  
class Subdivider:
  DENSIFY_COEF = 5
  MAX_ITER = 50
  
  def __init__(self, minarea=1, maxarea=sys.float_info.max, maxwagner=sys.float_info.max):
    self.minarea = minarea
    self.maxarea = maxarea
    self.maxwagner = maxwagner
    self.densifyDist = math.sqrt(self.maxarea) / float(self.DENSIFY_COEF)
    self.sideTolerance = self.densifyDist / float(self.DENSIFY_COEF)
    # import common
    # common.debug(self.minarea, self.maxarea, self.maxwagner)
    # global plt
    # import matplotlib.pyplot as plt
    # plt.figure().gca().set_aspect(True)
    # plt.ion()
    # plt.show()
  
  def subdivide(self, multipolygon):
    '''Returns a list of singlepart polygons with all the original polygon parts
    broken to pieces not larger than maxarea. Tries to achieve equal area sizes
    and short dividing lines.'''
    # preprocess the polygon (no multiparts, no holes)
    stack = multipolygon.explode() if multipolygon.isMultipart() else [multipolygon]
    for poly in stack:
      if not self.isOK(poly) and poly.hasHoles(0):
        poly.cutHoles()
    # we have unholy singleparts, start main
    while stack:
      # print([len(x.coors[0][0]) for x in stack])
      if len(stack) > 100:
        print(stack[-4].coors, stack[-4].area)
        print(stack[-3].coors, stack[-3].area)
        print(stack[-2].coors, stack[-2].area)
        print(stack[-1].coors, stack[-1].area)
        raise MemoryError
      current = stack.pop()
      # poly = multipolygon.coors[0][0]
      # xs = [pt[0] for pt in poly]
      # ys = [pt[1] for pt in poly]
      # plt.plot(xs, ys)
      # plt.plot(xs, ys, '.')
      # plt.plot([xs[0], xs[10]], [ys[0], ys[10]], '.')
      if self.isOK(current):
        current.sparsify()
        yield current
      else:
        current.densify(self.densifyDist)
        split = Splitter(current.coors[0][0], current.circumference, current.area, self.sideTolerance).find()
        if split:
          stack.extend(split)
        else:
          current.sparsify()
          yield current
        
  def isOK(self, poly):
    return poly.area < self.maxarea and (poly.area < self.minarea or poly.wagner < self.maxwagner)
    
  def subdivideList(self, coorlist):
    return [item.toList() for item in self.subdivide(MultiPolygon.fromList(coorlist))]
    # subs = self.subdivide(MultiPolygon.fromList(coorlist))
    # print([type(sub) for sub in subs])
    # return [item.toList() for item in subs]

class Splitter:
  LENGTH_WEIGHT = 0.5
  AREA_WEIGHT = 0.5
  IMPROVE_STEPS = 5
  SPLIT_DENSITY = 10
  VALIDATION_COEF = 10
  VALIDATION_SIZE_STEP = 5
  MAX_VALIDATION_STEPS = 4
  SIMILARITY_THRESHOLD = 3
  MAX_OFFSET = 10
  
  def __init__(self, ring, circumference, area, sidetol):
    self.fullRing = ring
    self.ring = MultiPolygon.generalize(self.fullRing, sidetol)
    self.translation = self.ringTranslation(self.ring, self.fullRing)
    self.length = len(self.ring) - 1
    self.idealHalf = self.length / 2
    self.targetMinSep = float(self.idealHalf / 2)
    self.circumference = circumference
    self.area = area
    self.convexity = vertexConvexity(self.ring)
    # print([(self.ring[i], i) for i in range(self.length) if self.convexity[i]])
    # print(self.convexity)
    self.lengthFactor = 2 * self.LENGTH_WEIGHT / self.circumference
    self.separationFactor = self.AREA_WEIGHT / self.idealHalf
    # self.areaFactor = 2 * self.AREA_WEIGHT
    self.lengthFactorExact = 2 / self.circumference
    self.areaFactorExact = 2
    self.generateCount = max(int(self.length / self.SPLIT_DENSITY), min(self.length, 5))
    self.generateStep = len(self.ring) / self.generateCount
    self.norm = numpy.linalg.norm
    self.improves = list(range(1, self.IMPROVE_STEPS+1))
    self.validationSize = max((2 * self.length - sum(self.convexity)) / self.VALIDATION_COEF, self.VALIDATION_SIZE_STEP) # earlier validation if more reflex vertices
    self.maxValidationSize = self.validationSize + self.VALIDATION_SIZE_STEP * self.MAX_VALIDATION_STEPS
    self.maxIter = self.length / self.SPLIT_DENSITY + self.IMPROVE_STEPS
    self.cost = self.costImprecise
    self.areaFactors = self.computeAreaFactors()
    # print(self.areaFactors)
    # if ((0.5 * abs(self.areaFactors.sum())) - self.area) > 1e-2:
      # print(0.5 * abs(self.areaFactors.sum()), self.area)
      # raise ValueError, 'area mismatch'
  
  @staticmethod
  def ringTranslation(ringFrom, ringTo):
    translation = []
    for i in range(len(ringTo)):
      if (ringFrom[len(translation)] == ringTo[i]).all():
        translation.append(i)
        if len(translation) == len(ringFrom):
          break
    return translation
  
  def areaFor(self, i, j):
    if i > j: i, j = j, i
    return 0.5 * abs(self.areaFactors[i:j].sum() + numpy.cross(self.fullRing[j], self.fullRing[i]))
  
  def costImprecise(self, i, j):
    return self.lengthFactor * self.norm(self.ring[i] - self.ring[j]) + \
        self.separationFactor * min(self.targetMinSep, abs(abs(i-j) - self.idealHalf))
  # return self.lengthFactor * self.norm(self.ring[i] - self.ring[j]) + \
        # self.separationFactor * abs(abs(i-j) - self.idealHalf)
  
  def costExact(self, i, j):
    # print(self.isValid((i,j)))
    i, j = self.translation[i], self.translation[j]
    subarea = self.areaFor(i,j)
    # print(subarea)
    # areaRatio = max(subarea, self.area - subarea) / self.area - 0.5
    # dividing into 1/4 and 3/4 is still perfectly OK
    # gives 0 for losers and 0.25 for owners
    areaRatio = 0.5 - min(subarea, self.area - subarea) / self.area
    # if (areaRatio - 0.5) > -COOR_EPSILON:
      # # print('too small')
      # return 1e9
    # print('exact', self.areaFactors[i:j].sum(), subarea, self.area, areaRatio, self.areaFactor * areaRatio, i, j)
    # print('exact', self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j]), self.areaFactorExact * areaRatio, 1 - (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])) ** self.LENGTH_WEIGHT * (1 - self.areaFactorExact * areaRatio) ** self.AREA_WEIGHT, i, j)
    # return self.lengthFactor * self.norm(self.fullRing[i] - self.fullRing[j]) + self.areaFactor * areaRatio
    # return 1 - (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])) ** self.LENGTH_WEIGHT * (1 - self.areaFactorExact * areaRatio) ** self.AREA_WEIGHT
    return 1 - (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])) ** self.LENGTH_WEIGHT * (1 - (self.areaFactorExact * areaRatio) ** 2) ** self.AREA_WEIGHT
  
  def move(self, candid, step=1):
    cost = self.cost
    length = self.length
    left = (candid[0], (candid[1] - step) % length)
    right = (candid[0], (candid[1] + step) % length)
    costLeft, costRight = cost(*left), cost(*right)
    return (left, costLeft) if costLeft < costRight else (right, costRight)
  
  def generate(self, offset=0):
    candidates = []
    for i in xrange(self.length):
      if not self.convexity[i]: # reflex vertex
        candidates.extend(self.generateFor(i, offset))
    # print('candidates', candidates)
    if not candidates: # convex
      for i in xrange(self.length):
        candidates.append((i, (i + self.idealHalf + offset) % self.length))
    # print(candidates)
    return candidates
    
  def generateFor(self, i, offset):
    for k in range(1, self.generateCount):
      yield (i, (int(k * self.generateStep) + i + offset) % self.length)
  
  def find(self, offset=0, validSize=None):
    split = None
    offset = 0
    validSize = self.validationSize
    while offset < self.MAX_OFFSET:
      if self.length < 4:
        return None
      candidates = self.generate(offset)
      candidates.sort(key=operator.itemgetter(0)) # sort by unmodifiable vertex index
      # print('candidates', candidates)
      iter = 0
      size = len(candidates)
      active = set(xrange(size))
      costs = [self.cost(*cand) for cand in candidates]
      # cycle until only one remains 
      # print('starting', self.maxIter, validSize)
      while len(active) > validSize and iter < self.maxIter:
        worst = None
        worstCost = 0.0
        for k in xrange(size):
          if k in active:
            # exclude similar worse splits
            p = k + 1
            maxx = candidates[k][0] + 3
            while p < size and candidates[p][0] < maxx:
              if p in active and self.similar(candidates[p], candidates[k]):
                if costs[p] < costs[k]:
                  active.remove(k)
                  break
                else:
                  active.remove(p)
              p += 1
            else:
              candidates[k], costs[k] = self.improve(candidates[k], costs[k])
              if costs[k] > worstCost:
                worst = k
                worstCost = costs[k]
        if worst is not None:
          active.remove(worst)
        iter += 1
      # select the best remaining (usually only one...)
      best = self.chooseBest(candidates, costs, self.validate(candidates, active))
      if best:
        return self.split(*self.climb(best))
      else:
        if validSize > self.maxValidationSize:
          offset += 1
          validSize = self.validationSize
        else:
          validSize += self.VALIDATION_SIZE_STEP
  
  def chooseBest(self, candidates, costs, active):
    best = None
    bestCost = 100.0
    # print([(candidates[i], costs[i]) for i in active])
    for i in active:
      if costs[i] < bestCost and self.isValid(candidates[i]):
        best = candidates[i]
        bestCost = costs[i]
    return best
  
  def similar(self, a, b):
    return (abs(a[0] - b[0]) + abs(b[1] - b[1])) < self.SIMILARITY_THRESHOLD
  
  def improve(self, split, cost, tabu=[]):
    for _ in self.improves:
      better, betterCost = self.move(split)
      # print('candidate', better, betterCost)
      if betterCost < cost and better not in tabu:
        split = better
        cost = betterCost
    return split, cost
  
  def climb(self, split):
    # raise RuntimeError
    self.cost = self.costExact
    # print('improving'self.cost(split))
    tabu = set()
    cost = self.cost(*split)
    # print('starting improvement', split, cost, self.isValid(split))
    for i in range(self.maxIter):
      newSplit, newCost = self.improve(split, cost, tabu=tabu)
      # print('considering', newSplit, newCost, tabu)
      if newSplit == split:
        subarea = self.areaFor(*(self.translation[split[0]], self.translation[split[1]]))
        # print('returning', split, cost, self.isValid(split), min(subarea, self.area - subarea))
        self.cost = self.costImprecise
        return split
      elif self.isValid(newSplit):
        split = newSplit
        cost = newCost
      else:
        tabu.add(newSplit)
        
  def validate(self, candidates, active):
    invalid = set()
    for i in active:
      if not self.isValid(candidates[i]):
        invalid.add(i)
    active -= invalid
    return active
  
  def isValid(self, split):
    trueSplit = (self.translation[split[0]], self.translation[split[1]])
    subarea = self.areaFor(*trueSplit)
    return edgeWithinRing(self.fullRing, *trueSplit) and min(subarea, self.area - subarea) > COOR_EPSILON
  
  def split(self, i, j):
    if i > j:
      i, j = j, i
    # print(self.isValid((i,j)), self.costExact(i,j))
    # print(self.splitInfo(i,j))
    # print(self.splitInfo(8,17))
    # print(self.splitInfo(17,8))
    subarea = self.areaFor(self.translation[i], self.translation[j])
    i = self.translation[i]
    j = self.translation[j]
    ring = self.fullRing
    # x = MultiPolygon([[ring[i:(j+1)] + [ring[i]]]])
    # y = MultiPolygon([[ring[:(i+1)] + ring[j:]]])
    # for p in x, y:
      # poly = p.coors[0][0]
      # xs = [pt[0] for pt in poly]
      # ys = [pt[1] for pt in poly]
      # plt.plot(xs, ys)
      # plt.plot(xs, ys, '.')
    # # print('parts printed')
    # z = raw_input()
    # print(i, j, x.area, y.area, self.areaFor(i,j))
    return (MultiPolygon([[ring[i:(j+1)] + [ring[i]]]]),
            MultiPolygon([[ring[:(i+1)] + ring[j:]]]))
  
  def splitInfo(self, i, j):
    print('SPLIT INFO', i, j)
    print('length', self.norm(self.ring[i] - self.ring[j]), 'x', self.circumference, 2*self.norm(self.ring[i] - self.ring[j])/self.circumference)
    print('separation', abs(abs(i-j) - self.idealHalf), 'x', self.idealHalf, min(self.targetMinSep, abs(abs(i-j) - self.idealHalf))/float(self.idealHalf))
    subarea = self.areaFor(self.translation[i], self.translation[j])
    areaRatio = min(0.25, min(subarea, self.area - subarea) / self.area)
    print('area', min(subarea, self.area - subarea), 'x', self.area, areaRatio)
    print('imprecise', self.lengthFactor * self.norm(self.ring[i] - self.ring[j]), self.separationFactor * min(self.targetMinSep, abs(abs(i-j) - self.idealHalf)), self.lengthFactor * self.norm(self.ring[i] - self.ring[j]) + self.separationFactor * min(self.targetMinSep, abs(abs(i-j) - self.idealHalf)))
    print('exact',
      (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])),
      (1 - self.areaFactorExact * areaRatio),
      1 - (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])) ** self.LENGTH_WEIGHT * (1 - self.areaFactorExact * areaRatio) ** self.AREA_WEIGHT)
    length = self.norm(self.fullRing[i] - self.fullRing[j])
    lenCrit = 2 * length / self.circumference
    areaCrit = 4 * max(0.0, 0.25 - min(subarea, self.area - subarea) / self.area)
    print('newexact', lenCrit, areaCrit, 1 - (1 - lenCrit) ** 0.5 * (1 - areaCrit) ** 0.5)
    print('real', self.costImprecise(i,j), self.costExact(i,j))
  
  # def costExact(self, i, j):
    # # print(self.isValid((i,j)))
    # i, j = self.translation[i], self.translation[j]
    # subarea = self.areaFor(i,j)
    # # print(subarea)
    # # areaRatio = max(subarea, self.area - subarea) / self.area - 0.5
    # # dividing into 1/4 and 3/4 is still perfectly OK
    # areaRatio = min(0.25, min(subarea, self.area - subarea) / self.area)
    # # if (areaRatio - 0.5) > -COOR_EPSILON:
      # # # print('too small')
      # # return 1e9
    # # print('exact', self.areaFactors[i:j].sum(), subarea, self.area, areaRatio, self.areaFactor * areaRatio, i, j)
    # # print('exact', self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j]), self.areaFactorExact * areaRatio, 1 - (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])) ** self.LENGTH_WEIGHT * (1 - self.areaFactorExact * areaRatio) ** self.AREA_WEIGHT, i, j)
    # # return self.lengthFactor * self.norm(self.fullRing[i] - self.fullRing[j]) + self.areaFactor * areaRatio
    # # return 1 - (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])) ** self.LENGTH_WEIGHT * (1 - self.areaFactorExact * areaRatio) ** self.AREA_WEIGHT
    # return 1 - (1 - self.lengthFactorExact * self.norm(self.fullRing[i] - self.fullRing[j])) ** self.LENGTH_WEIGHT * (1 - self.areaFactorExact * areaRatio) ** self.AREA_WEIGHT
  
  def computeAreaFactors(self):
    rgarr = numpy.array(self.fullRing)
    return rgarr[:-1,0] * rgarr[1:,1] - rgarr[:-1,1] * rgarr[1:,0]
  
def vertexConvexity(ring):
  return [bool(position(ring[i+1], ring[-2 if i == 0 else i-1], ring[i]) > -COOR_EPSILON) for i in xrange(len(ring)-1)]
  

if __name__ == '__main__':
  test = [[[[ 17.0203019,  49.4978009], [ 17.020304 ,  49.4977902], [ 17.0204787,  49.4974051], [ 17.0205789,  49.4973594], [ 17.0207696,  49.4973296], [ 17.0209131,  49.4973257], [ 17.0210167,  49.4973642], [ 17.0211046,  49.4974144], [ 17.0212231,  49.4974458], [ 17.02161  ,  49.4974653], [ 17.0216635,  49.4974779], [ 17.0217053,  49.4975476], [ 17.0218576,  49.4975497], [ 17.0218639,  49.4974922], [ 17.0219842,  49.4975039], [ 17.02198  ,  49.4977591], [ 17.02198  ,  49.4977594], [ 17.0215141,  49.4977663], [ 17.0211274,  49.4977892], [ 17.0204898,  49.4977927], [ 17.0203019,  49.497801 ], [ 17.0203019,  49.4978009]], [[ 17.0209961,  49.4976711], [ 17.0209877,  49.4977157], [ 17.0210632,  49.4977208], [ 17.0210729,  49.4976771], [ 17.0209961,  49.4976711]]]]
  # test = [[[[0,0], [5,0], [5,5], [0,5], [0,0]], [[1,1], [1,2], [2,1], [1,1]]]]
  # test = [[[[ 48.1984588,  16.3754003], [ 48.1985221,  16.3753936], [ 48.1988395,  16.375489 ], [ 48.1989129,  16.3755379], [ 48.1989633,  16.3755957], [ 48.1990014,  16.3756672], [ 48.1990277,  16.3757509], [ 48.1990398,  16.3758461], [ 48.1990257,  16.376026 ], [ 48.1989748,  16.3761809], [ 48.1989422,  16.3762467], [ 48.1989187,  16.3761949], [ 48.1981636,  16.3766203], [ 48.1981087,  16.3764042], [ 48.1981444,  16.3763845], [ 48.1980843,  16.3761527], [ 48.19805  ,  16.3761699], [ 48.1979954,  16.3759573], [ 48.1980893,  16.3758969], [ 48.1986157,  16.3755936], [ 48.1986157,  16.3755253], [ 48.1984588,  16.3754003]]]]
  # test = [[list(reversed([[ 48.2028652,  16.3737139], [ 48.2025198,  16.3739128], [ 48.202474 ,  16.3739391], [ 48.2024522,  16.3740204], [ 48.2024803,  16.3740749], [ 48.2027171,  16.3743235], [ 48.2027133,  16.3743317], [ 48.202855 ,  16.3744938], [ 48.2028621,  16.3744919], [ 48.2029777,  16.3746124], [ 48.2029811,  16.3746061], [ 48.2031941,  16.3748354], [ 48.2032309,  16.3747344], [ 48.2033696,  16.374431 ], [ 48.2034176,  16.374341 ], [ 48.2032723,  16.3741782], [ 48.2032745,  16.3741681], [ 48.2030121,  16.3738742], [ 48.2030092,  16.3738789], [ 48.2028652,  16.3737139]]))]]
  # test = [[[[  462040.41637864,  5547640.04785807], [  462010.6726797 ,  5547640.06597533], [  461980.92898077,  5547640.08409259], [  461951.18528184,  5547640.10220985], [  461921.4415829 ,  5547640.12032711], [  461891.69788397,  5547640.13844437], [  461891.67769733,  5547606.9973799 ], [  461891.65751069,  5547573.85631543], [  461891.63732405,  5547540.71525096], [  461891.61713742,  5547507.57418649], [  461891.59695078,  5547474.43312202], [  461899.39519756,  5547475.22305754], [  461899.53719756,  5547475.23535753], [  461914.89569756,  5547476.34975753], [  461947.76399756,  5547478.78085754], [  461995.82249756,  5547482.83965753], [  461995.92079756,  5547482.84695754], [  462030.31569757,  5547485.03719087], [  462032.33583378,  5547516.03932431], [  462034.35596999,  5547547.04145775], [  462036.37610621,  5547578.04359119], [  462038.39624242,  5547609.04572463], [  462040.41637864,  5547640.04785807]]]]
  # print(
  # print(MultiPolygon(test).area)
  # print(vertexConvexity([numpy.array(x) for x in test[0][0]]))
  # x = Subdivider(0.000000000001, 5e-8, 2.5).subdivideList(test)
  # x = Subdivider(2000, 20000, 2).subdivideList(test)
  # print(x)
  # for multipolygon in x:
    # poly = multipolygon[0][0]
    # xs = [pt[0] for pt in poly]
    # ys = [pt[1] for pt in poly]
    # plt.plot(xs, ys)
    # plt.plot(xs, ys, '.')
  # y = raw_input()
  # ingeom = {'type': 'Polygon', 'coordinates': [[(16.1364695, 50.353317), (16.1364576, 50.3532444), (16.1364445, 50.3531492), (16.137022, 50.353117), (16.137026, 50.353149), (16.137218, 50.3531395), (16.137242, 50.353272), (16.137238, 50.353275), (16.137161, 50.353275), (16.1364695, 50.353317)]]}
  # print(centroid(ingeom))
  print(test)
  print([x for x in MultiPolygon.fromList(test).triangulateSimple()])