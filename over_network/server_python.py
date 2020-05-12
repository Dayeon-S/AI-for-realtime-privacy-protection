import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import datetime

HOST="172.18.8.64"
PORT=8989

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

fcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(datetime.datetime.now().strftime("%d_%H-%M-%S.avi"), fcc, 25.0, (320,240))

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(1024)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(1024)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 0)
    print(frame)
    out.write(frame)
    cv2.imshow('ImageWindow',frame)
    cv2.waitKey(1)

out.release()
cv2.destroyAllWindows()
s.close()
