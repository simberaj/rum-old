from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

DEFAULT_ENCODING = 'utf8'

import attribute
import geojson
import itertools

class ConfigError(Exception):
  pass

class InputError(Exception):
  pass

class Transformer:  
  def __init__(self, config, outputFolder=None):
    self.layers = {}
    self.commonListeners = []
    self.geometryListeners = {geomtype : [] for geomtype in geojson.GEOMETRY_TYPES}
    self.editor = None
    self.configure(config)
    if outputFolder:
      self.setOutput(outputFolder)
  
  def setOutput(self, folder):
    # import arcpy
    for layer in self.layers.values():
      layer.setOutput(folder)
    # arcpy.env.overwriteOutput = True
    # self.editor = arcpy.da.Editor(folder)
  
  # def __enter__(self):
    # self.editor.startEditing(False, True) # no undo/redo stack, do commit at all times
    # self.editor.startOperation()
    # return self
  
  # def __exit__(self, exctype, excval, exctb):
    # self.editor.stopOperation()
    # self.editor.stopEditing(exctype is None)
    
    # print(self.commonListeners)
    # for geomtype in self.geometryListeners:
      # print(geomtype, self.geometryListeners[geomtype])
    # print(self.geometryListeners)

  def configure(self, config):
    try:
      if config['config'] != 'transformer':
        raise ConfigError, 'configuration mismatch: transformer expected, got ' + config['config']
      outputConfig = config['layers']
      for outputLayerName in outputConfig.iterkeys():
        layerConfig = outputConfig[outputLayerName]
        outputLayer = TransformLayer(layerConfig['geometry'], outputLayerName)
        if 'selector' in layerConfig:
          outputLayer.setSelector(attribute.selector(layerConfig['selector']))
        if 'attributes' in layerConfig:
          for attrConfig in layerConfig['attributes']:
            outputLayer.addAttribute(attribute.attribute(attrConfig))
        # for attr in outputLayer.attrs:
          # print(attr.name, attr.restrict)
        self.layers[outputLayerName] = outputLayer
        if 'input-geometries' in layerConfig:
          for ingeom in layerConfig['input-geometries']:
            for geojsonGeom in geojson.matchingGeometryTypes(ingeom):
              self.geometryListeners[geojsonGeom].append(outputLayer)
        else:
          for listenerList in self.geometryListeners.values():
            listenerList.append(outputLayer)
    except KeyError, mess:
      raise
    self.ids = set()
        
  def transform(self, feature):
    for layer in self.geometryListeners[feature['geometry']['type']]:
      for output in layer.transform(feature):
        yield output

        
class TransformLayer(object):
  def __init__(self, geomType, name):
    self.geomType = geomType
    self.name = name
    self.selector = None
    self.output = None
    self.attrs = []
    
  def setSelector(self, selector):
    self.selector = selector
  
  def addAttribute(self, attr):
    self.attrs.append(attr)
  
  def setOutput(self, folder):
    import common, loaders, os
    slots = {loaders.SHAPE_SLOT : loaders.SHAPE_SLOT}
    types = {}
    for attr in self.attrs:
      slots[attr.name] = attr.name
      types[attr.name] = attr.type
    self.output = loaders.GeoJSONBasedWriter(common.featurePath(folder, self.name), slots, types=types, shapeType=geojson.arcpyShapeType(self.geomType), crs=loaders.WGS_84_PRJ)
      
  def transform(self, feature):
    props = feature['properties']
    if self.selector is None or self.selector(props):
      # if props['osm_id'] == '207732211': print('trans-start', feature['geometry'])
      # if feature['geometry']['type'][-3:] == 'ing': 
        # print('trans-start', feature['geometry'])
      # if feature['geometry']['type'] != self.geomType:
        # print(self.name, self.geomType, feature['geometry']['type'])
      geometries = geojson.convertGeometryType(feature['geometry'], self.geomType)
      properties = self.transformProperties(props)
      # print(self.output)
      if properties:
        for geom in geometries:
          # if geom['type'][-3:] == 'ing': print('trans-end', geom)
          for props in properties:
            # if props['osm_id'] == '207732211': print('trans-end', geom)
            feature = geojson.feature(geom, props)
            # print(self.output, feature)
            if self.output:
              # print(self.name, feature)
              # print(feature)
              self.output.write(feature)
            yield feature
  
  def transformProperties(self, props):
    transformed = {}
    maxlen = None
    # print(self.attrs)
    for attr in self.attrs:
      value = attr.get(props)
      if isinstance(value, list):
        maxlen = len(value)
      elif value is None and attr.restrict:
        return None
      transformed[attr.name] = value
    if maxlen is None:
      return [transformed]
    else:
      return [{key : (value[i] if isinstance(value, list) else value) for key, value in transformed.iteritems()} for i in range(maxlen)]
      # except LookupError:
        # print(props)
        # print(transformed)
        # raise
      
  def __repr__(self):
    # return self.__class__.__name__ + '(' + unicode(self.selector) + ',[' + ','.join(unicode(at) for at in self.attrs) + '])'
    return self.__class__.__name__ + '(' + self.name + ')'
