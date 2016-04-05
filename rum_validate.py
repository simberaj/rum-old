import common
import rum
  
if __name__ == '__main__':
  with common.runtool(7) as parameters:
    parameters[5] = common.toBool(parameters[5], rum.ABSOLUTE_FIELD_DESC)
    parameters[6] = common.toBool(parameters[6], 'keep target switch')
    rum.validate(*parameters)
  