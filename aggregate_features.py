import features
import common

with common.runtool(2) as params:
  segments, points = params
  features.FeatureCalculator.aggregate(segments, points)