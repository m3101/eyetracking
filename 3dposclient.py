#This script should be run within the Blender python
#environment. It looks for two objects: "Camera", our camera,
#and "Camroot", which will represent our real camera reference IRL.
#The camera will be placed at the same displacement from the reference
#as the person is from the real camera

import numpy as np
import io
import bpy
import socket

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("./socketfile")
buffer = io.BytesIO()
def callupdate(scene):
    global sock
    buffer.seek(0)
    buffer.flush()
    sock.sendall(bytes([0]))
    l = sock.recv(1)
    buffer.write(sock.recv(l[0]))
    buffer.seek(0)
    pos_3d = np.load(buffer)
    root = bpy.data.objects['Camroot'].location
    cam = bpy.data.objects['Camera'].location
    cam.x=root.x+pos_3d[0]
    cam.y=root.y+pos_3d[1]
    cam.z=root.z+pos_3d[2]
    bpy.data.objects['Camera'].location=cam
bpy.app.handlers.frame_change_post.append(callupdate)