from __future__ import with_statement, print_function
from xml import sax
import json
import geojson
from collections import defaultdict
import geometry
import re
import os
import warnings
import gc
import numpy

MAX_TILE_SIZE_DEG = 0.1

class MalformedRelationError(Exception):
  pass

class BaseOSMHandler(sax.handler.ContentHandler):
  POLYGONAL_RELATIONS = set(('boundary', 'multipolygon'))
  SYSTEM_PROPS = ('source', 'fixme', 'created_by')
  REMOVE_PROPS = ['type']
  WINDINGS = {'outer' : False, 'inner' : True}
  MAX_SIZE = -1 # turn off, TODO: turn on
  
  def __init__(self, writer):
    sax.handler.ContentHandler.__init__(self)
    self.writer = writer
    self.featureCount = 0

  def writeFeature(self, geometry, properties, id=None):
    if (geometry and properties):
      if id is not None:
        properties['osm_id'] = id
      # if properties['osm_id'] == '207732211':
        # print('osm', geometry)
      self.writer.write(geometry, properties)
      self.featureCount += 1
    
  @classmethod
  def wayGeometry(cls, coors, tags=None):
    # try:
    if coors[0] == coors[-1] and not cls.hasPolylineTags(tags):# and not geometry.hasSelfIntersections(coors):
      if len(coors) > 3:
        geometry.setWinding(coors, False)
        return {'type' : 'Polygon', 'coordinates' : [coors]}
    else:
      return {'type' : 'LineString', 'coordinates' : coors}
    # except:
      # print(coors, wayrefs)
      # raise
  
  @staticmethod
  def hasPolylineTags(tags):
    if 'area' in tags:
      return tags['area'] == 'no' or ('highway' in tags and tags['highway'] == 'pedestrian')
    else:
      return ('highway' in tags or 'barrier' in tags)
      
  @classmethod
  def relationGeometry(cls, outer, inner, nodes):
    # role is True (outer rings) or False (inner rings)
    outerRings = cls.noderefsToRings(outer, nodes, False)
    if not outerRings: # degenerate outer rings only
      return None
    holeLists = cls.matchHoles(outerRings, cls.noderefsToRings(inner, nodes, True))
    coors = [[outerRings[i]] + holeLists[i] for i in xrange(len(outerRings))]
    return {'type' : 'MultiPolygon', 'coordinates' : coors}

  @classmethod
  def relationProperties(cls, props, wayprops):
    # print(props)
    # print(wayprops)
    allProps = props.copy()
    if wayprops:
      allWayProps = set(wayprops[0].iteritems())
      for wayPropDict in wayprops[1:]:
        allWayProps.intersection_update(wayPropDict.iteritems())
      # print(allWayProps)
      if allWayProps:
        allProps.update(dict(tuple(allWayProps)))
    # print(allProps)
    # print()
    # if len(wayPropSummary) == 1:
      # allProps.update(dict(wayPropSummary.pop()))
    for key in cls.REMOVE_PROPS:
      if key in allProps:
        del allProps[key]
    return allProps
    
  @classmethod
  def noderefsToRings(cls, noderefs, nodes, winding=False):
    # winding: False is counterclockwise
    # noderefs: list of sequences of node IDs
    # nodes: node ID: (lon, lat)
    rings = [cls.noderefsToLine(ring, nodes) for ring in cls.dropDegens(cls.ringify(noderefs))]
    while [] in rings:
      rings.remove([])
    for ring in rings:
      geometry.setWinding(ring, winding)
    return rings
    
  @staticmethod
  def dropDegens(ringlist):
    return [ring for ring in ringlist if len(ring) > 3]
    
  @staticmethod
  def noderefsToLine(wayrefs, nodes):
    return [nodes[ref] for ref in wayrefs]
  
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
          # if sum(len(way) for way in ways) < 1000:
            # print(ways)
          raise MalformedRelationError, 'open multipolygon' + ('' if id is None else (' for relation ' + str(id)))
    return rings
  
    
class OSMDirectHandler(BaseOSMHandler):
  def startDocument(self):
    self.nodeOn = False
    self.wayOn = False
    self.relOn = False
    self.nodes = {}
    self.nodeProps = defaultdict(dict)
    self.ways = defaultdict(list)
    self.wayProps = defaultdict(dict)
    self.relations = defaultdict(lambda: defaultdict(list))
    self.relProps = defaultdict(dict)
    self.last = None
    
  def startElement(self, name, attrs):
    if name == 'node':
      elID = int(attrs['id'])
      if elID not in self.nodes:
        self.nodeOn = True
        self.nodeID = int(attrs['id'])
        self.nodes[elID] = (float(attrs['lon']), float(attrs['lat']))
      if self.last == 'relation': # chunk ended
        self.writeWays()
    elif name == 'nd':
      if self.wayOn:
        self.ways[self.wayID].append(int(attrs['ref']))
    elif name == 'tag':
      if self.nodeOn:
        self.addTag(self.nodeProps[self.nodeID], attrs)
      elif self.wayOn:
        self.addTag(self.wayProps[self.wayID], attrs)
      elif self.relOn:
        self.addTag(self.relProps[self.relID], attrs)
    elif name == 'member':
      if self.relOn and attrs['type'] == 'way':
        self.relations[self.relID][attrs['role']].append(int(attrs['ref']))
    elif name == 'way':
      elID = int(attrs['id'])
      if elID not in self.ways:
        self.wayOn = True
        self.wayID = elID
    elif name == 'relation':
      elID = int(attrs['id'])
      if elID not in self.relations:
        self.relOn = True
        self.relID = elID
  
  def addTag(self, toDict, attrs):
    key = attrs['k']
    val = attrs['v']
    if key not in self.SYSTEM_PROPS:
      toDict[key] = (int(val) if val.isdigit() else val) # TODO: float values?
  
  def endElement(self, name):
    if name == 'node':
      if self.nodeID in self.nodeProps:
        self.writeNode(self.nodeID)
      self.nodeOn = False
    elif name == 'way':
      if self.wayID in self.ways:
        self.ways[self.wayID] = numpy.array(self.ways[self.wayID], dtype=numpy.dtype('u8'))
        # self.writeWay(self.wayID)
      self.wayOn = False
    elif name == 'relation':
      if self.relID in self.relProps:
        self.writeRelation(self.relID)
      if self.relID in self.relations:
        self.relations[self.relID] = None
      self.relOn = False
    self.last = name
  
  # def endDocument(self):
    # with open('e:\\temp\\dictdump', 'w') as file:
      # file.write(str(self.__dict__))
  
  def writeNode(self, id):
    self.writeFeature(
      {'type' : 'Point', 'coordinates' : self.nodes[id]},
      self.nodeProps[id],
      id)
    del self.nodeProps[id]

  def writeWays(self):
    for id in self.wayProps:
      self.writeWay(id)
    self.wayProps = defaultdict(dict)
    
  def writeWay(self, id):
    refs = self.ways[id]
    props = self.wayProps[id]
    self.writeFeature(self.wayGeometry(self.noderefsToLine(refs), props), props, id)
    del self.ways[id]
    # except:
      # print(id, refs, props)
      # raise
  
  def writeRelation(self, id):
    props = self.relProps[id]
    if 'type' in props and props['type'] in self.POLYGONAL_RELATIONS:
      refs = self.relations[id]
      props = self.relationProperties(refs, props)
      if props:
        try:
          geom = self.relationGeometry(refs['outer'] + refs[''], refs['inner'], self.nodes)
          # print('RELATION', geom)
          self.writeFeature(geom, props, id)
        except MalformedRelationError as message:
          warnings.warn(str(message) + ', skipping')
          # pass
          # print(props, refs)
          # raise
      del self.relations[id]
    del self.relProps[id]
  
 
class OSMDatabaseHandler(BaseOSMHandler):
  CREATE_SCHEMA = [
  ('nodegeoms', 'id integer primary key, lat real, lon real', ['id']),
  ('nodes', 'id integer primary key, tags text', ['id']),
  ('wayrefs', 'id integer, noderef integer, ord integer', ['id']),
  ('ways', 'id integer primary key, tags text', ['id']),
  ('relrefs', 'id integer, wayref integer, outer integer', ['id']),
  ('rels', 'id integer primary key, tags text', ['id'])]
  TABLE_NAMES = {'node' : 'nodes', 'way' : 'ways', 'relation' : 'rels'}

  def __init__(self, *args, **kwargs):
    BaseOSMHandler.__init__(self, *args, **kwargs)
    self.initDB()
  
  def initDB(self):
    import sqlite3
    self.dbfile = self.dbFileName()
    self.connection = sqlite3.connect(self.dbfile)
    self.connection.row_factory = sqlite3.Row
    self.duplicateError = sqlite3.IntegrityError
    self.cursor = self.connection.cursor()
    self.createTables()
  
  def createTables(self):
    for name, cols, indexCols in self.CREATE_SCHEMA:
      self.cursor.execute('create table {}({})'.format(name, cols))
      for col in indexCols:
        self.cursor.execute('create index {0}_{1} on {0}({1})'.format(name, col))
  
  @staticmethod
  def dbFileName():
    import random
    return os.path.join(os.getcwd(), 'tmp_' + str(int(random.random() * 1e10)))
    
  def startDocument(self):
    self.mode = None
    self.wayOrder = None
    self.currentTags = {}
  
  def startElement(self, name, attrs):
    try:
      if name == 'node':
        self.mode = name
        self.curID = int(attrs['id'])
        self.cursor.execute('insert into nodegeoms values (?,?,?)',
          (self.curID, float(attrs['lat']), float(attrs['lon'])))
      elif name == 'nd':
        if self.wayOrder:
          self.cursor.execute('insert into wayrefs values (?,?,?)',
            (self.curID, int(attrs['ref']), self.wayOrder))
          self.wayOrder += 1
      elif name == 'tag':
        key = attrs['k']
        val = attrs['v']
        if key not in self.SYSTEM_PROPS:
          self.currentTags[key] = (int(val) if val.isdigit() else val)
      elif name == 'member': # TODO
        if self.mode == 'relation' and attrs['type'] == 'way':
          self.cursor.execute('insert into relrefs values (?,?,?)',
            (self.curID, int(attrs['ref']), attrs['role'] != 'inner'))
      elif name in ('way', 'relation'):
        elID = int(attrs['id'])
        # if not self.exists(name, elID):
        self.wayOrder = 1
        self.mode = name
        self.curID = elID
    except self.duplicateError:
      pass # duplicate node/way/relation, pass it, no interest
  
  def endElement(self, name):
    if name in ('node', 'way', 'relation'):
      if self.currentTags and self.condition(name, self.currentTags):
        # print(self.currentTags)
        try:
          self.cursor.execute('insert into {} values (?,?)'.format(self.TABLE_NAMES[name]), (self.curID, json.dumps(self.currentTags)))
        except self.duplicateError:
          pass
        self.currentTags.clear()
      self.mode = None
      self.wayOrder = None
  
  def condition(self, name, tags):
    return name != 'relation' or ('type' in tags and tags['type'] in self.POLYGONAL_RELATIONS)
  
  def endDocument(self):
    self.connection.commit()
    self.writeNodes()
    self.writeWays()
    self.writeRelations()
    self.connection.close()
    os.unlink(self.dbfile)
  
  def writeNodes(self):
    cur = self.cursor
    cur.execute('select nodes.id as id, nodes.tags as tags, nodegeoms.lat as lat, nodegeoms.lon as lon from nodes join nodegeoms on nodes.id=nodegeoms.id')
    node = cur.fetchone()
    while node:
      self.writeFeature({'type' : 'Point', 'coordinates' : (node['lon'], node['lat'])}, json.loads(node['tags']), node['id'])
      node = cur.fetchone()
  
  def generate(self, name):
    cur = self.connection.cursor()
    cur.execute('select * from {}'.format(self.TABLE_NAMES[name]))
    item = cur.fetchone()
    while item:
      yield item
      item = cur.fetchone()
  
  def writeWays(self):
    cur = self.cursor
    for way in self.generate('way'):
      wayID = way['id']
      cur.execute('select nodegeoms.lat as lat, nodegeoms.lon as lon from nodegeoms join wayrefs on nodegeoms.id=wayrefs.noderef where wayrefs.id=? order by wayrefs.ord', (str(wayID), ))
      coors = [(pt['lon'], pt['lat']) for pt in cur.fetchall()]
      tags = json.loads(way['tags'])
      self.writeFeature(self.wayGeometry(coors, tags), tags, wayID)
    
  def writeRelations(self):
    cur = self.cursor
    nodes = {}
    for rel in self.generate('relation'):
      tags = json.loads(rel['tags'])
      relID = rel['id']
      cur.execute('''select nodegeoms.id as nodeid, nodegeoms.lat as lat, nodegeoms.lon as lon,
          wayrefs.id as wayid, ways.tags as tags, relrefs.outer as outer
        from ((relrefs join wayrefs on wayrefs.id=relrefs.wayref)
                join nodegeoms on nodegeoms.id=wayrefs.noderef)
                  left join ways on ways.id=relrefs.wayref
        where relrefs.id=?
        order by wayrefs.id, wayrefs.ord''', (str(relID), ))
      nodes.clear()
      waytags = set()
      wayrefs = {True : defaultdict(list), False : defaultdict(list)}
      for pt in cur.fetchall():
        waytags.add(pt['tags'])
        wayrefs[pt['outer']][pt['wayid']].append(pt['nodeid'])
        nodes[pt['nodeid']] = (pt['lon'], pt['lat'])
      # if relID == 2219956:
        # print(tags)
        # print(waytags)
        # print(self.relationProperties(tags, [json.loads(ts) for ts in waytags if ts]))
        # print(wayrefs)
        # print(nodes)
        # print(self.relationGeometry(wayrefs[True].values(), wayrefs[False].values(), nodes))
      try:
        relProps = self.relationProperties(tags, [json.loads(ts) for ts in waytags if ts])
        if relProps:
          relGeom = self.relationGeometry(wayrefs[True].values(), wayrefs[False].values(), nodes)
          if relGeom:
            self.writeFeature(relGeom, relProps, relID)
      except MalformedRelationError as message:
        warnings.warn(str(message) + ', skipping')

      
  # def exists(self, name, id):
    # print(name, id)
    # self.cursor.execute('select id from {} where id=?'.format(self.TABLE_NAMES[name]), (str(id), ))
    # return bool(self.cursor.fetchone())
     
     
def _process(inosm, listener):
  if os.path.getsize(inosm) < OSMDirectHandler.MAX_SIZE:
    handler = OSMDirectHandler(listener)
  else:
    handler = OSMDatabaseHandler(listener)
  parser = sax.make_parser()
  parser.setContentHandler(handler)
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
  
CHUNK_SIZE = 2048
    
def downloadTiles(tiles, target, prog=None):
  import urllib2
  # first, find out the tiles we need to download
  with open(target, 'w') as fileout:
    fileout.write(DOC_START.format(*allBox(tiles)))
    try:
      written = 0
      forechunk = ''
      for i in range(len(tiles)):
        # print('downloading', downloadAddress(tiles[i]))
        gate = urllib2.urlopen(downloadAddress(tiles[i]))
        chunk = gate.read(CHUNK_SIZE)
        first = True
        while chunk:
          if first:
            for regex in REMOVE:
              forechunk = regex.sub('', chunk)
              chunk = regex.sub('', chunk)
            first = False
          if forechunk:
            fileout.write(forechunk)
            written += 1
            if written > 1000:
              fileout.flush()
              written = 0
          forechunk = chunk
          chunk = gate.read(CHUNK_SIZE)
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
  print(extent)
  return 'http://overpass-api.de/api/interpreter?data=[timeout:720];(node({1},{0},{3},{2});<;>;);out%20meta;'.format(*tuple(str(x) for x in extent))
  # return 'http://overpass-api.de/api/map?bbox=' + ','.join(str(x) for x in extent)
 
if __name__ == '__main__':
  import sys, cProfile
  if len(sys.argv) > 3 and sys.argv[3].startswith('-d'):
    download(sys.argv[1], sys.argv[2])
  else:
    # cProfile.run('convert(sys.argv[1], sys.argv[2])')
    # print(sys.argv)
    convert(sys.argv[1], sys.argv[2])
  # 