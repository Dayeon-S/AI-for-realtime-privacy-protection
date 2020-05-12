import cv2
import socket
import io
import struct
import time
import pickle
import zlib

#use TCP
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("172.18.8.64", 8989))
connection = s.makefile('wb')

cam=cv2.VideoCapture(-1)

cam.set(3,320);
cam.set(4,240);

img_counter = 0

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
	ret,frame=cam.read()
	result, frame = cv2.imencode('.jpg', frame, encode_param)
	data = pickle.dumps(frame,0)
	size = len(data)

	print("{}: {}".format(img_counter,size))
	s.sendall(struct.pack(">L", size)+data)
	img_counter +=1

cam.release()
client_socket.clase()
