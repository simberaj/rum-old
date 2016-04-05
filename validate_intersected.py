import common
import rum

with common.runtool(4) as parameters:
  rum.validateIntersected(*parameters)