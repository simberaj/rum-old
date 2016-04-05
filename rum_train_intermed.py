import common
import rum
  
if __name__ == '__main__':
  with common.runtool(4) as parameters:
    rum.trainFromSegments(*parameters)
  