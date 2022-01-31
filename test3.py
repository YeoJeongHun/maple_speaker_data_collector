

from socket import *
from select import select
import sys

HOST = '127.0.0.1'
PORT = 8025
BUFSIZE = 1024

clientSocket = socket(AF_INET, SOCK_STREAM)

try:
    clientSocket.connect((HOST, PORT))
    clientSocket.sendall(bytes('hihihihihihihihi\n', 'UTF-8'))
    print('send\n')
    
    # data = clientSocket.recv(1024)
    # print(data.decode())
    clientSocket.close()
except Exception as e :
    print(e)