import common
import rum

with common.runtool(6) as parameters:
  parameters[5] = common.toBool(parameters[5], 'absolute/relative training value switch')
  rum.prepareTraining(*parameters)