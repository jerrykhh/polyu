import socket
import hashlib

SEGMENT_SIZE = 100

# try to send the message based on provided time (relay_time)
def __sendto(sock: socket.SocketType, msg:str, addr: tuple, relay_time: int) -> bool:

    tried_times = 0
    while tried_times < relay_time:
        try:
            sock.sendto(msg, addr)
            return True
        except socket.timeout:
            tried_times+=1
    return False
            
def udp_send(sock: socket.SocketType, addr: tuple, msg: str):
    # split the message if too large
    offset = 0
    seq = 0
    while offset < len(msg):
        if offset + SEGMENT_SIZE > len(msg):
            segment = msg[offset:]
        else:
            segment = msg[offset:offset + SEGMENT_SIZE]
        offset += SEGMENT_SIZE
        ack_received = False

        
        while not ack_received:
            
            # resend
            sent_status = __sendto(sock=sock, 
                                    msg=str(__udp_md5(segment) + str(seq) + segment).encode(), 
                                    addr= addr, relay_time=3)
            
            if sent_status:
                try:
                    # waiting send back message
                    mes_rec, addr_rec = sock.recvfrom(1024)
                    # decode message from binary to str
                    mes_rec = mes_rec.decode()
                except socket.timeout as e:
                    print("Timeouted!", e)

                else:
                    
                    # split the message to checksum + seq + content
                    
                    checksum = mes_rec[:32]
                    ack_seq = mes_rec[35]

                    # check information whether correct
                    
                    if __udp_md5(mes_rec[32:]) == checksum and ack_seq == str(seq):
                        ack_received = True # for leave the ack loop
        seq = 1 - seq

# md5 hash for message
def __udp_md5(mes: str) -> str:
    return str(hashlib.md5(mes.encode()).hexdigest())

# combine md5 hash and mes orgin message
def __udp_hash_mes(mes: str) -> str:
    return str(__udp_md5(mes) + mes).encode()

# receive message for udp
def udp_recv(sock: socket.SocketType, expecting_seq: int, connt: dict=None):
    while True:
        
        try:
            # when received message
            msg_rec, addr = sock.recvfrom(1024)
            # for update connection dict can be used to boardcast method
            if connt is not None:
                connt[addr] = addr
            
            # message decode
            msg_rec = msg_rec.decode()
            if len(msg_rec) == 0:
                continue
            # split the message
            checksum = msg_rec[:32]
            seq = msg_rec[32]
            content = msg_rec[33:]

            # check message whether correct
            if __udp_md5(content) == checksum:
                print(f"\nReceived a message and reutrn in uppercase: {content.upper()}")
                
                # send back to ack
                __sendto(sock=sock,
                            msg=__udp_hash_mes("ACK"+seq),
                            addr=addr,
                            relay_time=3)
                # update the seq
                if seq == str(expecting_seq):
                    expecting_seq = 1 - expecting_seq

            else:
                # cheange to negative_seq
                msg_exp_seq = f"ACK{str(1 - expecting_seq)}"
                __sendto(sock=sock, msg=__udp_hash_mes(msg_exp_seq), addr=addr, relay_time=3)
        except socket.timeout:
            pass
