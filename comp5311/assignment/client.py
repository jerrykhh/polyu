import argparse
import socket
import threading
import os
import time
from module import udp_send, udp_recv

# client tcp receive method
def tcp_recv(sock):
    while True:
        try:
            # when message receive
            msg_rec = sock.recv(1024).decode()
            print(f"\nReceive in uppercase: {msg_rec.upper()}")
        except ConnectionResetError or Exception:
            print("Server closed")
            sock.close()
            break
    os.close(0)

# client tcp send method
def tcp_send(sock):

    while True:
        msg = input("Please enter the test data to the server: ")
        # if input EXIT leave
        if msg == "EXIT":
            print("You close the Connection")
            sock.close()
            break
        # send message
        sock.send(msg.encode())
    os.close(0)

# client udp receive method
def udp_recv_proc(sock: socket.SocketType):
    expecting_seq = 0
    udp_recv(sock=sock, expecting_seq=expecting_seq)

# client udp send method
def udp_send_proc(host, port, sock: socket.SocketType):
    addr = (host, port)
    while True:
        msg = input("Please enter the test data to the server: ")
        # if input EXIT leave
        if msg == "EXIT":
            print("You close the Connection")
            sock.close()
            break
        # call udp_send method
        udp_send(sock, addr, msg)


def main(args):

    
    # check args.mode is udp or tcp
    if str(args.mode).lower() == 'udp':
        
        # create udp socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # set default timeout
        sock.settimeout(20)
        # new thread for receive message and send message
        threads = [threading.Thread(target=udp_recv_proc, args=[sock]), 
                   threading.Thread(target=udp_send_proc, args=(args.host, args.port, sock))]

    else:
        # check server connection
        connected = False
        while not connected:
            try:
                # create tcp socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # connect to server
                sock.connect((args.host, args.port))
                connected = True
            except ConnectionRefusedError:
                # if connect failed, reconnect after 1 secs
                print("Waiting Connection")
                time.sleep(1)
            
        
         # new thread for receive message and send message
        threads = [threading.Thread(target=tcp_recv, args=[sock]), threading.Thread(
            target=tcp_send, args=[sock])]
    # start thread
    for t in threads:
        t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP/UDP Server")
    parser.add_argument("-mode", type=str, default='tcp', help='tcp/udp')
    parser.add_argument("-host", type=str, default='127.0.0.1', help='Server Hostname/IP')
    parser.add_argument("-port", type=int, default=12349, help='Server Port')
    args = parser.parse_args()

    main(args=args)
