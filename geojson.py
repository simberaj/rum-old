import json
import geometry
import os

GEOMETRY_TYPES = ['Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']

SINGLEPART_TYPES = ['Point', 'LineString', 'Polygon']
MULTIPART_TYPES = ['MultiPoint', 'MultiLineString', 'MultiPolygon']

GEOMETRY_TYPES_TO_SIMPLE = {'Point' : 'point', 'MultiPoint' : 'point', 'LineString' : 'line', 'MultiLineString' : 'line', 'Polygon' : 'polygon', 'MultiPolygon' : 'polygon'}

SIMPLE_TYPES_TO_GEOMETRY = {'point' : ['Point', 'MultiPoint'], 'line' : ['LineString', 'MultiLineString'], 'polygon' : ['Polygon', 'MultiPolygon']}

MULTIPART_TO_SINGLEPART_TYPES = {'MultiPoint' : 'Point', 'MultiLineString' : 'LineString', 'MultiPolygon' : 'Polygon'}

ARCPY_SHAPE_TYPES = {'Point' : 'Point', 'MultiPoint' : 'MultiPoint', 'LineString' : 'Polyline', 'MultiLineString' : 'Polyline', 'Polygon' : 'Polygon', 'MultiPolygon' : 'Polygon'}

def arcpyShapeType(geojsonType):
  return ARCPY_SHAPE_TYPES[geojsonType]

def matchingGeometryTypes(intype):
  if intype in GEOMETRY_TYPES:
    return [intype]
  elif intype in SIMPLE_TYPES_TO_GEOMETRY:
    return SIMPLE_TYPES_TO_GEOMETRY[intype]
  else:
    raise KeyError, 'geometry type {} not recognized'.format(intype)

def convertGeometryType(geom, toType):
  fromType = geom['type']
  if fromType == toType:
    return [geom]
  elif fromType in MULTIPART_TYPES and MULTIPART_TO_SINGLEPART_TYPES[fromType] == toType:
    return [{'type' : toType, 'coordinates' : part} for part in geom['coordinates']]
  elif toType == 'Point':
    return [{'type' : 'Point', 'coordinates' : geometry.centroid(geom)}]
  else:
    raise ValueError, 'geometry retype failed: from {} to {}'.format(fromType, toType)
  
def feature(geometry, properties):
  return {'type' : 'Feature', 'geometry' : geometry, 'properties' : properties}
  
class Listener:
  def __init__(self, fields=[]):
    self.fields = fields

  def __enter__(self):
    return self
    
  def __exit__(self, *args):
    pass
    
  def processGeometry(self, geom):
    return geom
  
  def processProperties(self, props):
    for fld in self.fields:
      props.setdefault(fld) # inserts None in place of fields
    return props

  def writeFeature(self, feature):
    self.write(feature['geometry'], feature['properties'])
  
  def writeMoreFeatures(self, iter):
    for feat in iter:
      self.writeFeature(feat)
  
  def feature(self, geometry, properties):
    return feature(self.processGeometry(geometry), self.processProperties(properties))
      
class Writer(Listener):
  # TODO: implement float precision
  START = '{"type" : "FeatureCollection", "features" : ['
  END = ']}'
  JOINER = ','
  DEFAULT_ENCODING = 'utf8'
  
  def __init__(self, outPath, fields=[], precision=None, encoding=None):
    Listener.__init__(self, fields)
    self.outPath = outPath
    self.encoding = self.DEFAULT_ENCODING if encoding is None else encoding
    if precision is None: precision = 5
    self.floatFormatter = (u'%i' if precision == 0 else u'%.{}f'.format(precision))

  def __enter__(self):
    self.enter()
    return self
  
  def enter(self):
    self.file = open(self.outPath, 'w')
    self.file.write(self.START)
    self.count = 0
    return self
  
  def write(self, geometry, properties):
    if self.count:
      self.file.write(self.JOINER)
    self.file.write(json.dumps(self.feature(geometry, properties)).encode(self.encoding))
    self.count += 1
    if self.count % 1000 == 0:
      self.file.flush()
  
  def __exit__(self, *args):
    self.exit()
    
  def exit(self):
    self.file.write(self.END)
    self.file.close()

        
class Aggregator(Listener):
  def __init__(self, *args, **kwargs):
    Listener.__init__(self, *args, **kwargs)
    self.features = []
  
  def write(self, geometry, properties):
    self.features.append(self.feature(geometry, properties))
  
  def get(self):
    return self.features

    
class Bridge(Listener):
  def __init__(self, transformer, *args, **kwargs):
    Listener.__init__(self, *args, **kwargs)
    self.transformer = transformer
  
  def write(self, geometry, properties):
    for out in self.transformer.transform(self.feature(geometry, properties)):
      pass
      
class CompatibilityWriter(Writer):
  def write(self, feature):
    if self.count:
      self.file.write(self.JOINER)
    self.file.write(json.dumps(feature).encode(self.encoding))
    self.count += 1
    if self.count % 1000 == 0:
      self.file.flush()
    
      
class Reader:
  CHUNK_SIZE = 1024

  @classmethod
  def read(cls, file):
    start = -1
    # find the features list
    while start < 0:
      initial = file.read(cls.CHUNK_SIZE)
      if not initial:
        return
      start = initial.find('[')
    chunk = initial[(start+1):]
    lookpos = 0
    depth = 0
    left = chunk.find('{')
    right = chunk.find('}')
    final = False
    try:
      while True:
        if final or depth < 0:
          yield json.loads(chunk[:chunk.rfind(']')])
          break
        elif left < 0 or right < 0:
          while left < 0 or right < 0:
            prelen = len(chunk)
            chunk += file.read(cls.CHUNK_SIZE)
            postlen = len(chunk)
            if postlen == prelen:
              final = True
              break
            left = chunk.find('{', lookpos)
            right = chunk.find('}', lookpos)
            # print 'reading extra'
        # print left + start, right + start
        if left < right:
          depth += 1
          lookpos = left + 1
          # print 'down to', depth, 'at', left + start
          left = chunk.find('{', lookpos)
        else:
          depth -= 1
          lookpos = right + 1
          # print 'up to', depth, 'at', right + start
          right = chunk.find('}', lookpos)
          if depth == 0:
            yield json.loads(chunk[:lookpos])
            chunk = chunk[lookpos:].lstrip(',')
            # print(left + start, right + start, lookpos)
            left -= lookpos
            right -= lookpos
            # print(left + start, right + start, left, right, chunk)
            # start += lookpos
            lookpos = 0
            # print(left + start, right + start, left, right)
    except Exception as mess:
      print(chunk, lookpos, mess)
    
INCREMENTAL_SIZE = 50000000 # 50 MB as the maximum to fit in the memory at a time

def feed(jsonfile, transformer):
  with open(jsonfile) as infile:
    if os.path.getsize(jsonfile) > INCREMENTAL_SIZE:
      content = Reader.read(infile)
    else:
      content = json.load(infile)['features']
    for feat in content:
      for out in transformer.transform(feat):
        pass
        
def split(jsonfile, featcount=250000):
  i = 0
  part = 1
  pattern = '.part{}'.join(os.path.splitext(jsonfile))
  with open(jsonfile) as infile:
    content = Reader.read(infile)
    while True:
      with Writer(pattern.format(part)) as writer:
        for feat in content:
          writer.writeFeature(feat)
          i += 1
          if i > featcount:
            i = 0
            part += 1
            break
        else:
          return
    
 
    
class JSONTransformer:
  START = '{"type" : "FeatureCollection", "features" : ['
  END = ']}'
  JOINER = ','

  def __init__(self, layer, fields=[], precision=None):
    import arcpy
    self.layer = layer
    if not precision: precision = '2'
    self.floatFormatter = (u'%i' if precision == '0' else u'%.{}f'.format(precision))
    self.description = arcpy.Describe(self.layer)
    if hasattr(self.description, 'shapeFieldName'):
      self.shapeFld = self.description.shapeFieldName
      self.getGeometry = {u'Point' : arcpyToPoint, u'MultiPoint' : arcpyToMultiPoint, u'Polyline' : arcpyToLineString, u'Polygon' : arcpyToPolygon}[self.description.shapeType]
    else:
      self.shapeFld = None
    self.fields = fields
  
  def getFeatures(self):
    common.progress('opening layer')
    count = common.count(self.layer)
    cursor = arcpy.SearchCursor(self.layer)
    prog = common.progressor('converting', count)
    for row in cursor:
      now = {u'type' : u'Feature', u'properties' : self.getProperties(row)}
      if self.shapeFld is not None:
        now[u'geometry'] = self.getGeometry(row.getValue(self.shapeFld))
      yield now
      prog.move()
    prog.end()
  
  def getProperties(self, row):
    props = {}
    for field in self.fields:
      props[field] = row.getValue(field)
    return props
  
  def output(self, file, encoding='utf8'):
    file.write(self.START)
    start = True
    for featDict in self.getFeatures():
      if not start:
        file.write(self.JOINER)
      else:
        start = False
      file.write(self.toString(featDict).encode(encoding))
    file.write(self.END)
    
  
  # @staticmethod
  # def toString(object):
    # return json.dumps(object, separators=(',', ':')).replace("'", '"')

  # obsolete methods (fail for large datasets)
  def getDict(self):
    return {u'type' : 'FeatureCollection', u'features' : list(self.getFeatures())}
  
  def getString(self):
    return self.toString(self.getDict())
    
  def toString(self, object):
    if isinstance(object, str):
      return u'"' + object.decode('utf8') + u'"'
    elif isinstance(object, unicode):
      return u'"' + object + u'"'
    elif isinstance(object, float):
      return self.floatFormatter % object
    elif isinstance(object, list):
      return u'[' + u','.join([self.toString(sub) for sub in object]) + u']'
    elif isinstance(object, dict):
      return u'{' + u','.join([self.toString(key) + u':' + self.toString(value) for key, value in object.items()]) + u'}'
    elif object is None:
      return u'null'
    else:
      return unicode(str(object))

if __name__ == '__main__':
  import common
  with common.runtool(4) as parameters:
    layerName, target, fields, precision = parameters
    transformer = JSONTransformer(layerName, fields.split(';'), precision=precision)
    with open(target, 'w') as outfile:
      transformer.output(outfile)
