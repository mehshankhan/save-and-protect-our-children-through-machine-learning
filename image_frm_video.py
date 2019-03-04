import cv2
vidcap = cv2.VideoCapture('normal_1.mp4')
success,image = vidcap.read()
count = 1028
while success:
  img=cv2.resize(image,(171,128))
  cv2.imwrite("image%d.jpg" % count, img)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
