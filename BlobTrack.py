# BlobTrack1.py
# background subtraction and blob tracking experiments
# OpenCV / Python    2015-06-28 J.Beale
import cv2           # OpenCV version 3.0.0
import numpy as np   # Numpy version 1.9
import sys
import argparse
import time
from VideoHelper import VideoHelper

def cluster(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*
        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]
        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


# define kernels
kernel5 = np.array([[0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0]]).astype(np.uint8)

kernel3 =np.array([[0, 1, 0],
                   [1, 1, 1],
                   [1, 1, 0]]).astype(np.uint8)
                   
maxpoints = 1                                # maximum number of blobs to track at once   
vfilt = 0.2                                  # stepwise velocity filter factor (low-pass filter)  
maxvel = 15                                  # maximum physically likely velocity (delta-X per frame)
xdistThresh = 50                             # how many pixels must a blob travel, before it becomes an event?
xc = np.float32(np.zeros(maxpoints))         # x center coordinates  
yc = np.float32(np.zeros(maxpoints))         # y center coordinates                     
xo = np.float32(np.zeros(maxpoints))         # x center, previous frame
xvel = np.float32(np.zeros(maxpoints))       # x velocity, instantaneous
xvelFilt = np.float32(np.zeros(maxpoints))   # x velocity (filtered by rolling average)  
xstart = np.float32(np.zeros(maxpoints))     # x starting point (for distance-travelled) 
xdist = np.float32(np.zeros(maxpoints))      # x distance-travelled since starting point
font = cv2.FONT_HERSHEY_SIMPLEX              # for drawing text
frameset = []                                # set of frames that have action
framesetdata = {}                            # centroid points for action frames
framenumber = 0                              # keep track of the frame numbers
setnumber = 2                                # consecutive velocity events to crete a video segment
xsize=300
ysize=300
framespre=40
framespost=20
fps=30

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

                   
# open the video file
cap = cv2.VideoCapture(args["video"])

# get the frame size
frameExtractor = VideoHelper()
size = frameExtractor.getFrameSize(args["video"])
xproc = size[0]  # x,y resolution for processing is taken from the video file
yproc = size[1] 
record = True  # should we record video output?
voname = 'track-out1.avi' # name of video output to save
vonamerep = 'track-out2.avi'  # color track with circle

if (record):
    fourcc_video = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(voname, fourcc_video, fps, (xproc,yproc))  # open output video to record
    fourcc_videorep = cv2.VideoWriter_fourcc(*'XVID')
    videorep = cv2.VideoWriter(vonamerep, fourcc_videorep, fps, (xproc,yproc))  # open output video to record

history = 5
varThreshold = 18
detectShadows = True
fgbg = cv2.createBackgroundSubtractorMOG2(history,varThreshold,detectShadows)
# -----------------------------------------------------
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change blob detection thresholds
params.minThreshold = 200
params.maxThreshold = 255

params.minDistBetweenBlobs = 50

# Filter by Area.
params.filterByArea = True
params.minArea = 1000
params.maxArea = 5000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.02

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

while(1):
    framenumber +=1
    ret, frame = cap.read()

    if frame is None:   
        if (record):
            video.release()    
            videorep.release()
        cap.release()
        cv2.destroyAllWindows()        
        break

    #frame = cv2.resize(frame,(xproc,yproc))

    fgmask = fgbg.apply(frame)
    temp2 = cv2.erode(fgmask,kernel3,iterations = 2)    # remove isolated noise pixels with small kernel
    filtered = cv2.dilate(temp2,kernel5,iterations = 3) # dilate to join adjacent regions, with larger kernel
	
    inv = 255 - filtered  # invert black to white
    # Detect blobs.
    keypoints = detector.detect(inv)
    
    i = 0
    new = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR) # to allow us to draw colored circles on grey mask bitmap
    for k in keypoints:
        if (i < maxpoints):
          xc[i] = k.pt[0]   # x center of blob
          yc[i] = k.pt[1]   # y center of blob
          xs1 = int(k.pt[0]) # integer coords
          ys1 = int(k.pt[1])
          radius = int(k.size / 2)
          xvel[i] = xc[i] - xo[i]  # delta-x since previous frame
          if (abs(xvel[i]) > maxvel):  # force unreasonably large velocities (likely a glitch) to 0.0
            xvel[i] = 0
            xstart[i] = xc[i]  # reset x starting point to 'here'

          xdist[i] = xc[i] - xstart[i] # calculate distance this blob has travelled so far
          if abs(xvelFilt[i] - xvel[i]) < (2 + abs(xvelFilt[i]/2)):  # a sudden jump in value resets the average
            xvelFilt[i] = (vfilt * xvel[i]) + (1.0-vfilt)*xvelFilt[i]  # rolling average
          else:
            xvelFilt[i] = xvel[i]  # reset value without averaging
            val1 = abs(xvelFilt[i] - xvel[i])
            val2 = 2 + abs(xvelFilt[i]/2)
            
          #print "%d, %5.3f, %5.3f, %5.1f,  %5.2f, %5.0f" % (i, xc[i], yc[i], k.size, xvelFilt[i], xdist[i])
          tstring = "%5.2f" % (xvelFilt[i])
          tstring2 = "%4.0f" % (xdist[i])
          if (abs(xdist[i]) > xdistThresh) and (xs1 > 150) and (xs1 < (xproc-150)) and (ys1 > 150) and (ys1 < (yproc-150)):
            # assume the subject is in the center of the frame so ignore anything
            # that happens in the corners
            cv2.putText(new,tstring,(xs1-30,ys1+80), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(new,tstring2,(xs1-30,ys1+95), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.circle(new,(xs1,ys1),50,[0,0,255],2, cv2.LINE_AA)
            cv2.circle(frame,(xs1,ys1),50,[0,0,255],2, cv2.LINE_AA)
            x0 = int(xs1 - xsize/2)
            y0 = int(ys1 - ysize/2)
            x1 = int(xs1 + xsize/2)
            y1 = int(ys1 + ysize/2)
            cv2.rectangle(frame,(x0,y0),(x1,y1),(255,255,255),3) 
            print "%d, %5.3f, %5.3f, %5.1f,  %5.2f, %5.0f" % (i, xc[i], yc[i], k.size, xvelFilt[i], xdist[i])
            print "Setting velocity"
            frameset.append(framenumber)
            framesetdata[framenumber] = [xs1, ys1]

          new = cv2.circle(new,(xs1,ys1),radius,[0,50,255],2, cv2.LINE_AA)  # round blob 
          xo[i] = xc[i]   # remember current x-center value for next frame
          i += 1 
    
    # Draw detected blobs as red circles.
    if (record):
        video.write(new)                    # save frame of output video
        videorep.write(frame)
    cv2.imshow('source',frame)
    cv2.imshow("Keypoints", new)  # Show blob keypoints
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       if (record):
          video.release()
       cap.release()
       cv2.destroyAllWindows()
       break

if (record):
    video.release()
    videorep.release()
cap.release()
cv2.destroyAllWindows()


# find the top velocity events
maxgap=15
groups = cluster(frameset,maxgap)
framegroups = []

############################################
# test section to write out frames as videos
# extract the frames to be operated on
framegroupstest = []
videouttest = []
for set in groups:
   if len(set) >= setnumber:
      # process the video again extracting the relevant segments
      # input arguments are the frame numbers and location of the object in
      # the frame.
      segment = frameExtractor.extractVideoSegment(args["video"], set, framesetdata, framespre, framespost, xsize, ysize)
      framegroupstest.append(segment)
counter = 0
for frameg in framegroupstest:
    # write to a file
    counter = counter + 1
    voname= "tracktest" + str(counter) + ".avi"
    fourcc_pre = cv2.VideoWriter_fourcc(*'XVID')
    outvideo = cv2.VideoWriter(voname, fourcc_pre, fps, (xproc,yproc))  # open output video to record
    for frame in frameg:
        outvideo.write(frame)
    outvideo.release()
# new write just the segment for this set
framegroupstest = []
videouttest = []
for set in groups:
   print "test: processing frame set: " + str(set)
   if len(set) >= setnumber:
      # process the video again extracting the relevant segments
      # input arguments are the frame numbers and location of the object in
      # the frame.
      segment = frameExtractor.extractMovingWindowVideoSegment(args["video"], set, framesetdata, framespre, framespost,xsize,ysize)
      framegroupstest.append(segment)
counter = 0
for frameg in framegroupstest:
    # write to a file
    counter = counter + 1
    voname= "tracktestwindow" + str(counter) + ".avi"
    fourcc_prewindow = cv2.VideoWriter_fourcc(*'XVID')
    outvideo = cv2.VideoWriter(voname, fourcc_prewindow, fps, (xproc,yproc))  # open output video to record
    for frame in frameg:
        newframe = frameExtractor.initVideoFrame(args["video"])
        newframe[0:frame.shape[0], 0:frame.shape[1]] = frame
        outvideo.write(newframe)
    outvideo.release()
# test section to write out frames as videos
############################################

# extract the frames to be operated on
for set in groups:
   if len(set) >= setnumber:
      # process the video again extracting the relevant segments
      # input arguments are the frame numbers and location of the object in 
      # the frame.
      segment = frameExtractor.extractMovingWindowVideoSegment(args["video"], set, framesetdata, framespre, framespost, xsize, ysize)
      framegroups.append(segment)

# write a video with the framegroups      
print "starting to record"
numframes = framespre + framespost + 60
loc = [[100,50],[100,370],[100,690],[420,50],[420,370],[420,690]]
videoout = []
for i in range(1,numframes):   # init frames
    frame = frameExtractor.initVideoFrame(args["video"])
    videoout.append(frame) 
counter=0
for frameg in framegroups:
    print "frameg is size " + str(len(frameg))
    startpoint = loc[counter]
    x = startpoint[0]
    y = startpoint[1]
    counter = counter + 1 
    print "this is x " + str(x)
    print "this is y " + str(y)
    for i in range(0,len(frameg)):
        data = frameg[i]
        temp = videoout[i]
        temp[x:x+data.shape[0], y:y+data.shape[1]] = data
        videoout[i] = temp
    if counter == 6:
        break
# add a title and date
for i in range(len(videoout)):
    frame = videoout[i]
    eventstring = "Hintertux Training"
    timestring = str(time.strftime("%d/%m/%Y"))
    cv2.putText(frame,eventstring,(10,50), font, 2,(255,255,255),4,cv2.LINE_AA)
    cv2.putText(frame,timestring,(10,100), font, 1,(255,255,255),4,cv2.LINE_AA)
    videoout[i] = frame

# write to a file
voname= "track.avi"
fourcc_pre = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter(voname, fourcc_pre, fps, (xproc,yproc))  # open output video to record
for frame in videoout:
    outvideo.write(frame)
# write the original video
#cap = cv2.VideoCapture(args["video"])
#while(1):
#    ret, frame = cap.read()
#    if frame is None:
#       break
#    outvideo.write(frame)
#cap.release()
outvideo.release()
