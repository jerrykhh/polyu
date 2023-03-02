import socket
import threading
import argparse
import abc
from module import udp_send, udp_recv

# abstract server for tcp/udp
class socketSERVER(abc.ABC):
    
    def __init__(self, host, port, socket_AF: socket.AddressFamily, socket_kind: socket.SocketKind) -> None:
        self.addr = (host, port)
        self.server_socket = socket.socket(socket_AF, socket_kind)
        self.connt = {}
        
    def start(self):
        self.server_socket.bind(self.addr)
        print("Server started, waiting the connection")
    
    # send message to all client
    def broadcast(self, msg: str):
        for sock in self.connt.values():
            sock.send(msg.encode())
            
    # close all socket and stop server
    def close(self):
        for sock in self.connt.values():
            sock.close()
        self.server_socket.close()
        
    @abc.abstractmethod
    def __accept_connection(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def __recv(self):
        raise NotImplementedError()
        

class tcpSERVER(socketSERVER):
    
    def __init__(self, host: str, port: int) -> None:
        super().__init__(host, port, socket.AF_INET, socket.SOCK_STREAM)
        
    def start(self):
        print("tcp ", end='')
        super().start()
        # set connection to be 1 client
        self.server_socket.listen(1)
        self._socketSERVER__accept_connection()
    
    def _socketSERVER__accept_connection(self):
        # Accept the connection
        client_socket, addr = self.server_socket.accept()
        # save the connection socket
        self.connt[addr] = client_socket
        # new thread for receive new message
        threading.Thread(target=self._socketSERVER__recv, args=(client_socket, addr)).start()
        print("Client connected.")
    
    def _socketSERVER__recv(self, client_socket: socket.SocketType, addr):
        while True:
            try:
                # when received message
                msg_rec:str = client_socket.recv(1024).decode().strip()
                # check message vaild or invaild
                if(len(msg_rec) == 0):
                    continue
                print(f"\nReceived a message and reutrn in uppercase: {msg_rec.upper()}")
            except ConnectionResetError:
                print("Connection Close")

class udpSERVER(socketSERVER):
    
    def __init__(self, host, port) -> None:
        # create the udp socket
        super().__init__(host, port, socket.AF_INET, socket.SOCK_DGRAM)
        self.expecting_seq = 0
        self.server_socket.settimeout(20)
        
    def start(self):
        super().start()
        self._socketSERVER__accept_connection()
        
    # send message to all user
    def broadcast(self, msg: str):
        for addr in self.connt.values():
            udp_send(self.server_socket, addr, msg)
    
    # new thread for recevie the new message
    def _socketSERVER__accept_connection(self):
        threading.Thread(target=self._socketSERVER__recv).start()
    
    def _socketSERVER__recv(self):
        udp_recv(
            sock=self.server_socket,
            expecting_seq=self.expecting_seq,
            connt=self.connt)


def main(args):
    # create tcp/udp server object
    if str(args.mode).lower() == 'udp':
        server = udpSERVER(host=args.host, port=args.port)
    else:
        server = tcpSERVER(host=args.host, port=args.port)
    
    # server start
    server.start()
    
    while True:
        msg = input("Please enter the test data to the client: ")
        # send message to client
        server.broadcast(msg)
        
        # exit
        if msg == "EXIT":
            server.close()
            break
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP/UDP Server")
    parser.add_argument("-mode", type=str, default='tcp', help='tcp/udp')
    parser.add_argument("-host", type=str, default='127.0.0.1', help='Server Hostname/IP')
    parser.add_argument("-port", type=int, default=12349, help='Server Port')
    args = parser.parse_args()
    main(args)