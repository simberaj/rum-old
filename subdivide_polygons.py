import sys
import common
import loaders
import geometry

MIN_RATIO = 5

def _subdivided(features, maxarea, maxwagner, minarea=None):
  prog = common.progressor('subdividing features', len(features))
  shapeSlot = loaders.SHAPE_SLOT
  for feat in features:
    for sub in geometry.subdivide(feat[shapeSlot],
        maxarea / MIN_RATIO if not minarea else minarea, maxarea, maxwagner):
      item = feat.copy()
      item[shapeSlot] = sub
      yield item
    prog.move()
  prog.end()

def subdivide(source, target, maxarea, maxwagner, minarea, transferFlds):
  minarea = float(minarea) if minarea else None
  maxarea = float(maxarea) if maxarea else sys.float_info.max
  maxwagner = float(maxwagner) if maxwagner else sys.float_info.max
  shapeSlot = loaders.SHAPE_SLOT
  inSlots = {shapeSlot : None}
  outSlots = {shapeSlot : shapeSlot}
  for fld in transferFlds:
    inSlots[fld] = fld
    outSlots[fld] = fld
  ld = loaders.BasicReader(source, inSlots)
  subs = list(_subdivided(ld.read(text='reading features'), maxarea, maxwagner, minarea))
  # import cProfile
  # cProfile.runctx("subs = list(_subdivided(ld.read(text='processing'), maxarea, maxwagner, minarea))", globals(), locals())
  wr = loaders.BasicWriter(target, outSlots, crs=source)
  wr.write(subs)

if __name__ == '__main__':
  with common.runtool(6) as parameters:
    source, target, maxarea, maxwagner, minarea, transferFldsStr = parameters
    transferFlds = common.parseFields(transferFldsStr)
    subdivide(source, target, maxarea, maxwagner, minarea, transferFlds)
    # import cProfile
    # cProfile.run('subs = main(ld)')
    # subs = []
    # for feat in ld.read(text='processing'):
      # for sub in geometry.subdivide(feat[shapeSlot], maxarea / MIN_RATIO, maxarea, maxwagner):
        # item = feat.copy()
        # item[shapeSlot] = sub
        # subs.append(item)
    