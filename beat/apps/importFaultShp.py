#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:22:36 2021

@author: Marin Govorcin
"""

import numpy as np
import os, sys
import argparse

class faultSegments:
    def __init__(self, segmentPointsLonLat= None, segmentPointsXY= None, segmentMidPoints= None, segmentStrike= None, segmentLength= None):
        if segmentPointsLonLat is None:
            self.segmentPointsLonLat = []
        else:
            self.segmentPointsLonLat = segmentPointsLonLat
        if segmentPointsXY is None:
            self.segmentPointsXY = []
        else:
            self.segmentPointsXY = segmentPointsXY
        if segmentMidPoints is None:
            self.segmentMidPoints = []
        else:
            self.segmentMidPoints = segmentMidPoints
        if segmentStrike is None:
            self.segmentStrike = []
        else:
            self.segmentStrike = segmentStrike
        if segmentLength is None:
            self.segmentLength = []
        else:
            self.segmentLength = segmentLength
            
    def addPointsLonLat(self,data):
        self.segmentPointsLonLat.append(data)
           
    def addPointsXY(self,data):
        self.segmentPointsXY.append(data)
                  
    def addMidPoints(self,data):
        self.segmentMidPoints.append(data)
    
    def addStrikes(self,data):
        self.segmentStrike.append(data)
    
    def addLengths(self,data):
        self.segmentLength.append(data)
    
    def plotFault(self):
        from matplotlib import pyplot as plt
        
        #plot fault segments
        z=1
        for i in range(len(self.segmentPointsXY)):
            for j in range(len(self.segmentPointsXY[i])):
                plt.plot(self.segmentPointsXY[i][j,:].reshape(2,2)[0],self.segmentPointsXY[i][j,:].reshape(2,2)[1],'-k')
                plt.plot(self.segmentMidPoints[i][j,0],self.segmentMidPoints[i][j,1],'bo')
                plt.text(self.segmentMidPoints[i][j,0]+100,self.segmentMidPoints[i][j,1]+100, 'F{} , Strike: '.format(z) + str(int(self.segmentStrike[i][j]))+'°')
                z = z +1 
        #plot event location
        plt.plot(0,0,'r+', label = 'Event location',ms=14)
        plt.xlabel("East shift [m]")
        plt.ylabel("North shift [m]")
        plt.title(" Fault segements to model in BEAT")
        

def import_faultShp(daShapefile):
    try:
        from osgeo import ogr
        print('Import of ogr from osgeo worked!\n')
    except:
        print('Import of ogr from osgeo failed\n\n')
    
    # import data from shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data = driver.Open(daShapefile,0)
    
    # Check to see if shapefile is found.
    if data is None:
        print('Could not open %s' % daShapefile)
    else:
        print('Opened %s' % daShapefile)
        layer = data.GetLayer()
        featureCount = layer.GetFeatureCount()
        print("Number of features in %s: %d" % (os.path.basename(daShapefile),featureCount))
        
    #extract Fault Point Coordinates from shapefile
    faultPoints = {}
    
    for i in range(featureCount):
        feature         = layer.GetFeature(i)
        featureGeometry = feature.GetGeometryRef()
        fnPoints        = featureGeometry.GetPointCount()
        featurePoints   = np.zeros((fnPoints,2))
        
        #extract points from shapefile features
        for j in range(fnPoints): 
            featurePoints[j,:] = np.asarray(featureGeometry.GetPoint(j)[0:2])
            
        faultPoints[i] = featurePoints
        
    # Output array of fault points
    return faultPoints


def getFaultSegments(event, faultSource):
    from pyrocko import orthodrome
    #Get fault points coordinates in local coordinate system with respect to the event location
    
    # Get point coordinates from shapefile 
    # event :: list [lon, lat]
    # faultSource :: str "filename.shp"
    fault = import_faultShp(faultSource)
    
    fSegments = faultSegments()
    
    #get Points from fault features - lines
    for i in range(len(fault)):
        faultXY = orthodrome.latlon_to_ne_numpy(fault[i][:,1],fault[i][:,0],event[1],event[0]) 
        
        fSegmentPointsLonLat = np.zeros((len(faultXY[0])-1,4))
        fSegmentPointsXY = np.zeros((len(faultXY[0])-1,4))
        fSegmentMidPoint = np.zeros((len(faultXY[0])-1,2))
        fSegmentStrike = np.zeros((len(faultXY[0])-1,1))
        fSegmentLength = np.zeros((len(faultXY[0])-1,1))
        
        
        for j in range(len(faultXY[0])-1):
            fSegmentPointsLonLat[j,:] = [fault[i][j][0], fault[i][j+1][0], fault[i][j][1], fault[i][j+1][1]] 
            fSegmentPointsXY[j,:] = [faultXY[1][j]*-1, faultXY[1][j+1]*-1, faultXY[0][j]*-1, faultXY[0][j+1]*-1] # x1 x2 y1 y2
            fSegmentMidPoint[j,:] = [(faultXY[1][j] + faultXY[1][j+1])/-2, (faultXY[0][j] + faultXY[0][j+1])/-2]
            fSegmentStrike[j,:]   = orthodrome.azimuth(fault[i][j,1],fault[i][j,0],fault[i][j+1,1],fault[i][j+1,0])
            fSegmentLength[j,:]   = orthodrome.distance_accurate15nm(fault[i][j,1],fault[i][j,0],fault[i][j+1,1],fault[i][j+1,0])
            
        fSegments.addPointsLonLat(fSegmentPointsLonLat)  
        fSegments.addPointsXY(fSegmentPointsXY)
        fSegments.addMidPoints(fSegmentMidPoint) #[East_shift, North_shift]
        fSegments.addStrikes(fSegmentStrike)
        fSegments.addLengths(fSegmentLength)
                
    return fSegments

def mfaultToBeat(faultSegments):
    # default lower upper testvalue
     priors = ['depth', 'dip','duration', 'east_shift', 'length', 'north_shift', 'nucleation_x', 'nucleation_y',  'rake', 'slip', 'strike', 'time', 'width']
     default_bounds = {
            'rake'         : [0, 180, 90],
            'dip'          : [0, 90,  45],
            'depth'        : [0, 30,  15],
            'slip'         : [0, 5,  2.5],
            'width'        : [0, 10,   5],
            'duration'     : [0, 30,  15],
            'nucleation_x' : [-1, 1,   0],
            'nucleation_y' : [-1, 1,   0],
            'time'         : [-5, 5,   0]}
     
     file=open('beatPriors_multiSegment.txt','w')
     
     midPoints   = np.vstack(faultSegments.segmentMidPoints)
     east_shift  = midPoints[:,0] / 1000 # in km
     north_shift = midPoints[:,1] / 1000 # in km
     strike      = np.vstack(faultSegments.segmentStrike)
     length      = np.vstack(faultSegments.segmentLength) / 1000 # in km
       
     bounds = {'east_shift':east_shift, 'north_shift':north_shift, 'strike': strike, 'length': length}
    # Print the beat.prior config file
    
     print('  priors:',file=file)
     for prior in priors:
         print('    %s: !beat.heart.Parameter' % prior,file=file)
         print('      name: %s' % prior,file=file)
         print('      form: Uniform',file=file)
         
         print('      lower:',file=file)
         if prior in default_bounds:
             for z in range(len(east_shift)):
                 print('      - %.2f' % default_bounds[prior][0],file=file)
         elif prior in bounds:
             for z in range(len(east_shift)):
                 print('      - %.2f' % bounds[prior][z],file=file)
         else:
             print('Error')
             
         print('      upper:',file=file)
         if prior in default_bounds:
             for z in range(len(east_shift)):
                 print('      - %.2f' % default_bounds[prior][1],file=file)
         elif prior in bounds:
             for z in range(len(east_shift)):
                 print('      - %.2f' % bounds[prior][z],file=file)
         else:
             print('Error')
             
         print('      testvalue:',file=file)
         if prior in default_bounds:
             for z in range(len(east_shift)):
                 print('      - %.2f' % default_bounds[prior][2],file=file)
         elif prior in bounds:
             for z in range(len(east_shift)):
                 print('      - %.2f' % bounds[prior][z],file=file)
         else:
             print('Error')

######################################################################33
def create_parser():
    parser = argparse.ArgumentParser(description=f'Get BEAT prior config for multiple fault segments from shapefile',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('file', type=str, help='shapefile of fault segments, e.g. "fault.shp"')
    parser.add_argument('-elon', '--event_lon', dest='event_lon', type=str, required=True,
                        help='event longitude')
    parser.add_argument('-elat', '--event_lat', dest='event_lat', type=str, required=True,
                        help='event latitude')
    return parser

def cmd_line_parser(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    return inps

##############################################################################
def main(iargs=None):
    inps = cmd_line_parser(iargs)
    
    # Get fault segements
    event = [float(inps.event_lon), float(inps.event_lat)]
    fault = getFaultSegments(event, inps.file)
    
    # Export BEAT config file
    mfaultToBeat(fault)
    print('Writing BEAT prior config in beatPriors_multiSegment.txt')


##############################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
