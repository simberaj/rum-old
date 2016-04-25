import common
import rum

with common.runtool(8) as parameters:
  parameters[7] = common.toBool(parameters[7], 'absolute/relative training value switch')
  rum.dualPrepareTraining(*parameters)