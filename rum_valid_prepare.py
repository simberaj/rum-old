import common
import rum

with common.runtool(6) as parameters:
  parameters[-1] = common.toBool(parameters[-1], 'absolute/relative disaggregation value switch')
  rum.prepareValidation(*parameters)