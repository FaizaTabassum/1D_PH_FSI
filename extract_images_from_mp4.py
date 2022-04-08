import cv2
vidcap = cv2.VideoCapture(r'C:\Users\Faiza\Desktop\1D_PH_FSI\videos\Medien1.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("C:/Users/Faiza/Desktop/1D_PH_FSI/videos/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
