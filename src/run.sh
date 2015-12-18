#!/bin/bash
time python FeatureTransformer.py --d temp
time python runlibsvm.py --d temp --steps scale,pred,evaluation