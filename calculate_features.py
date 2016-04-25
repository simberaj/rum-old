import common
import rum

with common.runtool(4) as parameters:
  parameters[3] = common.split(parameters[3])
  rum.calculateFeatures(*parameters)