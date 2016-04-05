import common
import rum
  
if __name__ == '__main__':
  with common.runtool(6) as parameters:
    parameters[5] = common.toBool(parameters[5], rum.ABSOLUTE_FIELD_DESC) # absolute?
    rum.applyToSegments(*parameters)
  