import arcpy
import loaders

if __name__ == '__main__':
  with common.runtool(7) as parameters:
    lines, typeFld, configFile, builtup, bufferDist, builtupMinSize, urbanFld, outfile = parameters
    with common.PathManager(outfile) as pathman:
      intravilan = pathman.tmpFile()
      bufferIntravilan(builtup, bufferDist, builtupMinSize, intravilan, shout=True)
      calculateSpeeds(lines, typeFld, loaders.jsonToConfig(configFile), intravilan, outfile, urbanFld, shout=True)
      
def bufferIntravilan(builtup, bufferDist, builtupMinSize, intravilan, shout=False):
  with common.PathManager(intravilan) as pathman:
    # buffer the inputs and dissolve them all together
    if shout: common.progress('buffering built-up areas')
    buffered = pathman.tmpFile()
    arcpy.Buffer_analysis(builtup, buffered, bufferDist, 'FULL', 'ROUND', 'ALL')
    if shout: common.progress('detecting small buffers')
    sizeNumber, sizeUnit = builtupMinSize.split()
    sizeFld = pathman.tmpField(buffered, float)
    arcpy.CalculateField_management(buffered, sizeFld, 
      '!shape.area@{}!'.format(sizeUnit), 'PYTHON')
    if shout: common.progress('delineating intravilan')
    arcpy.Select_analysis(buffered, intravilan,
      common.safeQuery('[{}] < {}'.format(sizeFld, sizeNumber), buffered))
    
def calculateSpeeds(lines, typeFld, config, intravilan, urbanFld=None, shout=False):
  if shout: common.progress('identifying intravilan lines')
  # transfer only FID to the identified result
  arcpy.Identity_analysis(lines, intravilan, outfile, 'ONLY_FID')
  if shout: common.progress('calculating speeds')
  # TODO: mark by function
  builtupFidFld = 'FID_' + os.path.splitext(os.path.split(intravilan)[1])[0]
  arcpy.AddField_management(outfile, urbanFld, common.pyTypeToOut(int))
  arcpy.CalculateField_management(outfile, urbanFld, 
    '0 if !{}! == -1 else 1'.format(builtupFidFld), 'PYTHON', speedFx)
  # if urban field specified, calculate it
  if urbanFld:
    arcpy.AddField_management(outfile, urbanFld, common.pyTypeToOut(int))
    arcpy.CalculateField_management(outfile, urbanFld, 
      '0 if !{}! == -1 else 1'.format(builtupFidFld), 'PYTHON')
  arcpy.DeleteField_management(outfile, builtupFidFld)