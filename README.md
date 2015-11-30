# actionvideointro


Intro video was joined with source video with the following
commands:

ffmpeg -i track.avi -qscale:v 1 intermediate1.mpg
ffmpeg -i s.mp4 -qscale:v 1 intermediate2.mpg
ffmpeg -i concat:"intermediate1.mpg|intermediate2.mpg" -c copy intermediate_all.mpg
ffmpeg -i intermediate_all.mpg -qscale:v 2 final.avi
