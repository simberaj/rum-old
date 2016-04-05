import common
import osm
import osm_to_direct_layers
import os
import loaders


def osmFilePath(folder, clipping):
  datasetName = os.path.splitext(os.path.basename(clipping))[0]
  return os.path.join(folder, datasetName + '.osm')

if __name__ == '__main__':
  with common.runtool(2) as parameters:
    clipping, targetOSM = parameters
    common.progress('determining tiles to download')
    tiles = list(loaders.findExtentTiles(clipping, osm.MAX_TILE_SIZE_DEG, osm.MAX_TILE_SIZE_DEG))
    common.message('Found {} tiles to download.'.format(len(tiles)))
    osm.downloadTiles(tiles, targetOSM, prog=common.progressor('downloading tiles', len(tiles)))