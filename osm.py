from __future__ import with_statement, print_function
from xml import sax
import json
import geojson
from collections import defaultdict
import geometry
import re
import warnings

MAX_TILE_SIZE_DEG = 0.1

class MalformedRelationError(Exception):
  pass

class OSMHandler(sax.handler.ContentHandler):
  POLYGONAL_RELATIONS = set(('boundary', 'multipolygon'))
  SYSTEM_PROPS = ('type', 'source', 'fixme', 'created_by')
  WINDINGS = {'outer' : False, 'inner' : True}

  def __init__(self, writer):
    sax.handler.ContentHandler.__init__(self)
    self.writer = writer
  
  def writeFeature(self, geometry, properties, id=None):
    if geometry is not None:
      for prop in self.SYSTEM_PROPS:
        if prop in properties:
          del properties[prop]
      if not properties:
        return
      if id is not None:
        properties['osm_id'] = id
      # if properties['osm_id'] == '207732211':
        # print('osm', geometry)
      self.writer.write(geometry, properties)

  def startDocument(self):
    self.featureCount = 0
    self.nodes = {}
    self.nodeProps = defaultdict(dict)
    self.ways = defaultdict(list)
    self.wayProps = defaultdict(dict)
    self.relations = defaultdict(lambda: defaultdict(list))
    self.relProps = defaultdict(dict)
    self.nodeOn = False
    self.wayOn = True
    
  def startElement(self, name, attrs):
    if name == 'node' and attrs['id'] not in self.nodes:
      self.nodeOn = True
      self.nodeID = attrs['id']
      self.nodes[self.nodeID] = (float(attrs['lon']), float(attrs['lat']))
    elif name == 'nd':
      if self.wayOn:
        self.ways[self.wayID].append(attrs['ref'])
    elif name == 'tag':
      if self.nodeOn:
        self.addTag(self.nodeProps[self.nodeID], attrs)
      elif self.wayOn:
        self.addTag(self.wayProps[self.wayID], attrs)
      elif self.relOn:
        self.addTag(self.relProps[self.relID], attrs)
    elif name == 'member':
      if self.relOn and attrs['type'] == 'way':
        self.relations[self.relID][attrs['role']].append(attrs['ref'])
    elif name == 'way' and attrs['id'] not in self.ways:
      self.wayOn = True
      self.wayID = attrs['id']
    elif name == 'relation' and attrs['id'] not in self.relations:
      self.relOn = True
      self.relID = attrs['id']
  
  @staticmethod
  def addTag(toDict, attrs):
    val = attrs['v']
    toDict[attrs['k']] = (int(val) if val.isdigit() else val) # TODO: float values?
  
  def endElement(self, name):
    # TODO: some attributes go to geometry, if so, ignore them in feature creation
    if name == 'node':
      if self.nodeID in self.nodeProps:
        self.writeNode(self.nodeID)
      self.nodeOn = False
    elif name == 'way':
      # if self.wayID in self.wayProps:
        # self.writeWay(self.wayID)
      self.wayOn = False
    elif name == 'relation':
      if self.relID in self.relProps: # TODO: what if of multiple ways with the same tags?
        self.writeRelation(self.relID)
      self.relOn = False
  
  def endDocument(self):
    for id in self.wayProps:
      self.writeWay(id)
  
  def writeNode(self, id):
    self.writeFeature(
      {'type' : 'Point', 'coordinates' : self.nodes[id]},
      self.nodeProps[id],
      id)
    del self.nodeProps[id]

  def writeWay(self, id):
    refs = self.ways[id]
    props = self.wayProps[id]
    try:
      self.writeFeature(self.wayGeometry(refs, props), props, id)
    except:
      print(id, refs, props)
      raise
  
  def writeRelation(self, id):
    props = self.relProps[id]
    if 'type' in props and props['type'] in self.POLYGONAL_RELATIONS:
      refs = self.relations[id]
      props = self.relationProperties(refs, props)
      if props:
        try:
          geom = self.relationGeometry(refs, props, id)
          # print('RELATION', geom)
          self.writeFeature(geom, props, id)
        except MalformedRelationError as message:
          warnings.warn(str(message) + ', skipping')
          # pass
          # print(props, refs)
          # raise
    del self.relProps[id]
  
  def relationProperties(self, refs, props):
    for prop in self.SYSTEM_PROPS:
      if prop in props:
        del props[prop]
    if not props:
      propVars = []
      for role in refs:
        for ref in refs[role]:
          propVars.append(self.wayProps[ref])
      # print(propVars)
      propVars = set(tuple(var.items()) for var in propVars if var)
      # print(propVars)
      if len(propVars) == 1:
        for ref in refs[role]:
          del self.wayProps[ref] # remove the ways that gave the props to the relation
        return dict(propVars.pop())
    return props
  
  def wayGeometry(self, wayrefs, tags=None):
    coors = self.noderefsToLine(wayrefs)
    try:
      if coors[0] is coors[-1] and not self.hasPolylineTags(tags):# and not geometry.hasSelfIntersections(coors):
        if len(coors) > 3:
          geometry.setWinding(coors, False)
          return {'type' : 'Polygon', 'coordinates' : [coors]}
      else:
        return {'type' : 'LineString', 'coordinates' : coors}
    except:
      print(coors, wayrefs)
      raise
  
  def relationGeometry(self, relrefs, tags=None, id=None):
    # print('refs', relrefs)
    # print('reftr', len(self.ways), [self.ways[id] for id in relrefs['outer'] if id in self.ways])
    relrefs['outer'].extend(relrefs[''])
    rings = {}
    for role in self.WINDINGS.keys():
      noderefs = [self.ways[id] for id in relrefs[role] if id in self.ways]
      # print(noderefs)
      rings[role] = [self.noderefsToLine(ring) for ring in self.ringify(noderefs, id)]
      for ring in rings[role]:
        geometry.setWinding(ring, self.WINDINGS[role])
      # for ring in rings[role]:
        # if geometry.hasSelfIntersections(ring):
          # print('self intersection found')
          # raise MalformedRelationError
    # print('rings', rings)
    outer = rings['outer']
    holeLists = self.matchHoles(outer, rings['inner'])
    coors = [self.dropDegens([outer[i]] + holeLists[i]) for i in xrange(len(outer))]
    while None in coors:
      coors.remove(None) # degenerate outer rings
    if coors:
      return {'type' : 'MultiPolygon', 'coordinates' : coors}
    else:
      raise MalformedRelationError, 'hole matching failed' + ('' if id is None else (' for relation ' + str(id)))
    
  def dropDegens(self, ringlist):
    if len(ringlist[0]) > 3:
      return [ring for ring in ringlist if len(ring) > 3]
    else:
      return None
    
  def noderefsToLine(self, wayrefs):
    return [self.nodes[ref] for ref in wayrefs]
  
  @staticmethod
  def matchHoles(outer, inner): # match inner to outer rings
    if len(outer) == 1:
      return [inner]
    else:
      holes = [[] for i in xrange(len(outer))]
      for hole in inner:
        onept = hole[0]
        for i in range(len(outer)):
          if geometry.pointInPolygon(onept, outer[i]):
            holes[i].append(hole)
            break
      return holes
        # we leave out holes whose first point is not inside
  
  @staticmethod
  def ringify(ways, id=None): # merges ways to rings
    rings = []
    i = 0
    while i < len(ways):
      if not ways[i]:
        ways.pop(i)
      elif ways[i][0] == ways[i][-1]: # closed ways are parts on their own
        rings.append(ways.pop(i))
      else:
        j = i + 1
        while j < len(ways):
          if ways[i][-1] == ways[j][0]: # succesor found
            ways[i] = ways[i][:-1] + ways.pop(j)
            break
          elif ways[i][0] == ways[j][-1]: # predecessor
            ways[i] = ways.pop(j) + ways[i][1:]
            break
          elif ways[i][0] == ways[j][0]: # reverse predecessor
            ways[i] = list(reversed(ways.pop(j))) + ways[i][1:]
            break
          elif ways[i][-1] == ways[j][-1]: # reverse successor
            ways[i] = ways[i][:-1] + list(reversed(ways.pop(j)))
            break
          else:
            j += 1
        else:
          if sum(len(way) for way in ways) < 1000:
            print(ways)
          raise MalformedRelationError, 'open multipolygon' + ('' if id is None else (' for relation ' + str(id)))
    return rings
  
  @staticmethod
  def hasPolylineTags(tags):
    if 'area' in tags:
      return tags['area'] == 'no' or ('highway' in tags and tags['highway'] == 'pedestrian')
    else:
      return ('highway' in tags or 'barrier' in tags)
      
def _process(inosm, listener):
  parser = sax.make_parser()
  parser.setContentHandler(OSMHandler(listener))
  parser.parse(inosm)

def convert(inosm, outgeojson, encoding=None):
  with geojson.Writer(outgeojson, encoding=encoding) as writer:
    _process(inosm, writer)

def extract(inosm, listener):
  _process(inosm, listener)
    
def parse(inosm):
  with geojson.Aggregator() as aggregator:
    _process(inosm, aggregator)
    return aggregator.get()

REMOVE = [re.compile(expr) for expr in [
  r'^\<\?xml.*?\?\>',
  r'\<osm.*?\>',
  r'(?m)\<note\>.*?\<\/note\>',
  r'\<meta.*?\/\>',
  r'\<bounds.*?\/\>',
  r'\<\/osm.*?\>$'
  ]]
  
DOC_START = '''<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="Overpass API via PyOSM download">
<note>The data included in this document is from www.openstreetmap.org. The data is made available under ODbL.</note>
<bounds minlon="{}" minlat="{}" maxlon="{}" maxlat="{}" />'''

DOC_END = '</osm>'
    
def download(clipping, target):
  import loaders
  tileCoors = list(loaders.findExtentTiles(clipping, MAX_TILE_SIZE_DEG, MAX_TILE_SIZE_DEG))
  return downloadTiles(tileCoors, target)
  
    
def downloadTiles(tiles, target, prog=None):
  import urllib2
  # first, find out the tiles we need to download
  with open(target, 'w') as fileout:
    fileout.write(DOC_START.format(*allBox(tiles)))
    try:
      for i in range(len(tiles)):
        # print('downloading', downloadAddress(tiles[i]))
        gate = urllib2.urlopen(downloadAddress(tiles[i]))
        doc = gate.read()
        for regex in REMOVE:
          doc = regex.sub('', doc)
        fileout.write(doc)
        if prog is not None: prog.move()
    finally:
      fileout.write(DOC_END)
    if prog is not None: prog.end()

def allBox(bboxes):
  return (min(bbox[0] for bbox in bboxes),
          min(bbox[1] for bbox in bboxes),
          max(bbox[2] for bbox in bboxes),
          max(bbox[3] for bbox in bboxes))
          
      
def downloadAddress(extent): # s w n e
  return 'http://overpass-api.de/api/interpreter?data=(node({1},{0},{3},{2});<;>;);out%20meta;'.format(*tuple(str(x) for x in extent))
  # return 'http://overpass-api.de/api/map?bbox=' + ','.join(str(x) for x in extent)
 
if __name__ == '__main__':
  import sys, cProfile
  # cProfile.run('convert(sys.argv[1], sys.argv[2])')
  convert(sys.argv[1], sys.argv[2])
  # download(sys.argv[1], sys.argv[2])