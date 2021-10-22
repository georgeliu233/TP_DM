import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("0.0.0.0", 33086))
# stuff here
s.close() # this is how you close the socket