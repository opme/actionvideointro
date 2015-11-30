# class for extracting video segment from a larger video file
import cv2           # OpenCV version 3.0.0
import numpy as np   # Numpy version 1.9
import sys

class VideoHelper:
    def __init__(self):
        pass

    #
    # extract a video segment 
    # arguments:
    # videofile - path to the video file
    # set - set of frames that contain the action of interest
    # framesetdata - all interesting frames along with the xy point of interest
    # xsize - total x pixels to grab
    # ysize - total y pixels to grab
    # framepre - number of frames to take before the average of this set of interesting frames
    # framepost - number of frames to take after from average of this set of interesting frames
    # 
    def extractVideoSegment(self, videofile, set, framesetdata, framespre, framespost, xsize, ysize):
        frameset = []
        framenumber = 0
     
        # calculate the interesting frames
        framefirst = set[0]
        framelast = set[-1] 
        framestart = framefirst - framespre
        if framestart < 1:
           framestart = 1  # don't let starting frame be less then frame 1
        frameend = framelast + framespost
 
        print "framefirst " + str(framefirst)
        print "framelast " + str(framelast)
        print "framestart " + str(framestart)
        print "frameend " + str(frameend)

        # open the video file
        cap = cv2.VideoCapture(videofile)
        while(1):
            # skip to the first interesting frame
            framenumber +=1
            if framenumber > frameend:
               cap.release()
               return frameset
            ret, frame = cap.read()
            if frame is None:
                cap.release()
                print "This is a limited frameset"
                return frameset
            if framenumber < framestart:
                continue
            # find the coordinates of interest and make a window around it
            coordinate = framesetdata.get(framenumber)
            if coordinate == None:
                # did not find coordinate for this frame, use the closest
                if framenumber < framefirst:
                    coordinate = framesetdata[framefirst]
                    print "test: coordinate from frame first " + str(coordinate)
                elif framenumber > framelast:
                    coordinate = framesetdata[framelast]
                    print "test: coordinate from frame last " + str(coordinate)
                else:
                    # take the closest coordinate
                    coordinate = framesetdata.get(framenumber, framesetdata[min(framesetdata.keys(), key=lambda k: abs(k-framenumber))])
                    print "test: coordinate from framesetdata else " + str(coordinate)
            else:
                coordinate = framesetdata.get(framenumber)
                print "test: coordinate from framesetdata " + str(coordinate)
            xcenter = coordinate[0]
            ycenter = coordinate[1]
            radius = 50
            cv2.circle(frame,(xcenter,ycenter),radius,[0,0,255],2, cv2.LINE_AA)
            x0 = int(xcenter - xsize/2)
            y0 = int(ycenter - ysize/2)
            x1 = int(xcenter + xsize/2)
            y1 = int(ycenter + ysize/2)
            cv2.rectangle(frame,(x0,y0),(x1,y1),(255,255,255),3)
            frameset.append(frame)

    #
    # extract a moving window from within a video segment
    # arguments:
    # videofile - path to the video file
    # set - set of frames that contain the action of interest
    # framesetdata - all interesting frames along with the xy point of interest
    # xsize - total x pixels to grab
    # ysize - total y pixels to grab
    # framepre - number of frames to take before the average of this set of interesting frames
    # framepost - number of frames to take after from average of this set of interesting frames
    #
    def extractMovingWindowVideoSegment(self, videofile, set, framesetdata, framespre, framespost, xsize, ysize):
        frameset = []
        framenumber = 0

        # get the size of the video
        framesize = self.getFrameSize(videofile)
        xproc = framesize[0]
        yproc = framesize[1]
        chan = framesize[2]


        # calculate the interesting frames
        framefirst = set[0]
        framelast = set[-1]
        framestart = framefirst - framespre
        if framestart < 1:
           framestart = 1  # don't let starting frame be less then frame 1
        frameend = framelast + framespost

        print "framefirst " + str(framefirst)
        print "framelast " + str(framelast)
        print "framestart " + str(framestart)
        print "frameend " + str(frameend)

        # open the video file
        cap = cv2.VideoCapture(videofile)
        while(1):
            # skip to the first interesting frame
            framenumber +=1
            if framenumber > frameend:
               cap.release()
               return frameset
            ret, frame = cap.read()
            if frame is None:
                cap.release()
                print "This is a limited frameset"
                return frameset
            if framenumber < framestart:
                continue
            # find the coordinates of interest and make a window around it
            coordinate = framesetdata.get(framenumber)
            if coordinate == None:
                # did not find coordinate for this frame, use the closest
                if framenumber < framefirst:
                    coordinate = framesetdata[framefirst]
                    print "coordinate from frame first " + str(coordinate)
                elif framenumber > framelast:
                    coordinate = framesetdata[framelast]
                    print "coordinate from frame last " + str(coordinate)
                else:
                    # take the closest coordinate
                    coordinate = framesetdata.get(framenumber, framesetdata[min(framesetdata.keys(), key=lambda k: abs(k-framenumber))])
                    print "coordinate from framesetdata else " + str(coordinate)
            else:
                coordinate = framesetdata.get(framenumber)
                print "coordinate from framesetdata " + str(coordinate)
            xcenter = coordinate[0]
            ycenter = coordinate[1]
            x0 = int(xcenter - xsize/2)
            y0 = int(ycenter - ysize/2)
            x1 = int(xcenter + xsize/2)
            y1 = int(ycenter + ysize/2)
            print "initial bounding box: " + str(x0) + "," + str(y0) + " " + str(x1) + "," + str(y1)
            # can not be less then 0
            if x0 < 0:
                x1 = int(x1 - x0)
                x0 = 0 
                print "adjusted x0 since less then 0"
            if y0 < 0:
                y1 = int(y1 - y0)
                y0 = 0
                print "adjusted y0 since less then 0"
            if x1 > xproc:
                x1 = int(x1 - (x1 - xproc))
                x0 = int(x0 - (x1 - xproc))
                print "adjusted x1 since greater then " + str(xproc)
            if y1 > yproc:
                y1 = int(y1 - (y1 - yproc))
                y0 = int(y0 - (y1 - yproc)) 
                print "adjusted y1 since greater then " + str(yproc)
            # now crop the frame
            print "bounding box: " + str(x0) + "," + str(y0) + " " + str(x1) + "," + str(y1)
            crop = frame[y0:y1, x0:x1]
            frameset.append(crop)


    #
    # extract a moving window from a frame and return a list
    # of the new frames
    # arguments:
    # segment - set of frames in a list
    # set - frames that we have x,y data for in this series
    # framesetdata - all interesting frames along with the xy point of interest
    # xsize - total x pixels to grab
    # ysize - total y pixels to grab
    #
    def getWindowFromFrame(self, segment, set, framsetdata, xsize, ysize):
       windowset = []
       # we need to extract a smaller video frame from the data
       print "getWindowFromFrame xsize: " + str(xsize)
       print "getWindowFromFrame ysize: " + str(ysize)
       print set
       framenumber = 0
       for frame in segment:
          framenumber += 1
          # if we do not have segment data for this frame
          # then use the data from the closest segment 
          #if framesetdata[framenumber] == None:
          crop = frame[0:xsize, 0:ysize]
          cv2.imshow("Crop",crop)
          windowset.append(crop)
       return windowset

    #
    # determine the size of a video file
    # arguments:
    # videofile - path to the video file
    #
    def getFrameSize(self, videofile):
        framesize = []
        # open the video file
        cap = cv2.VideoCapture(videofile)
        ret, frame = cap.read()
        if frame is None:
            ys = 0
            xs = 0
            chan = 0
        ys,xs,chan = frame.shape
        cap.release()
        framesize.append(xs)
        framesize.append(ys)
        framesize.append(chan)
        return framesize 

    #
    # init a blank video frame
    # arguments:
    # videofile - path to the video file used to init 
    #
    def initVideoFrame(self, videofile):
        cap = cv2.VideoCapture(videofile)
        ret, frame = cap.read()
        ysp,xsp,chan = frame.shape
        cv2.rectangle( frame, ( 0,0 ), ( xsp, ysp), ( 0,0,0 ), -1, 8 )
        cap.release()
        return frame
