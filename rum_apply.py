import common
import rum

if __name__ == '__main__':
  with common.runtool(8) as parameters:
    parameters[7] = common.toBool(parameters[7], rum.ABSOLUTE_FIELD_DESC) # absolute?
    rum.applyModel(*parameters)