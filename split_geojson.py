import geojson
import sys

if len(sys.argv) > 2:
  geojson.split(sys.argv[1], int(sys.argv[2]))
else:
  geojson.split(sys.argv[1])