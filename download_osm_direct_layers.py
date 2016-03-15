import common
import osm
import osm_to_direct_layers
import os
import loaders


def osmFilePath(folder, clipping):
  datasetName = os.path.splitext(os.path.basename(clipping))[0]
  return os.path.join(folder, datasetName + '.osm')

if __name__ == '__main__':
  with common.runtool(3) as parameters:
    clipping, workspace, config = parameters
    common.progress('determining tiles to download')
    folder = common.folder(workspace)
    targetOSM = osmFilePath(folder, clipping)
    tiles = list(loaders.findExtentTiles(clipping, osm.MAX_TILE_SIZE_DEG, osm.MAX_TILE_SIZE_DEG))
    common.message('Found {} tiles to download.'.format(len(tiles)))
    osm.downloadTiles(tiles, targetOSM, prog=common.progressor('downloading tiles', len(tiles)))
    common.progress('converting OSM data')
    osm_to_direct_layers.convert(targetOSM, workspace, config, prj=clipping, clip=clipping)
