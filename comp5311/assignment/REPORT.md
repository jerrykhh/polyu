# COMP5311 Assignment 1

## Usage
```
usage: client.py [-h] [-mode MODE!] [-host HOST] [-port PORT]
usage: server.py [-h] [-mode MODE!] [-host HOST] [-port PORT]
```

```bash
# server
python server.py -mode tcp
# client
python client.py -mode tcp
```

## Code Design

#### `client.py`

This file is implementation of a Python client that can communicate with the server over UDP and TCP connections. The client can send messages to the server, and the server converts them to uppercase letters and sends back the response. 

##### Usage
```bash
python client.py -mode tcp
# or
python client.py -mode udp
```

The client module consists of the following methods:
- `tcp_recv`: Receives messages from the server over TCP and prints them in uppercase letters.
- `tcp_send`: Accepts user input messages and sends them to the server over TCP.
- `udp_recv_proc`: Receives messages from the server over UDP and prints them in uppercase letters.
- `udp_send_proc`: Accepts user input messages and sends them to the server over UDP.
- `main`: The main method that sets up the client and initializes the appropriate methods for the selected mode (UDP/TCP).

For user can input the message and receives the message from the server. Threading is appiled.

#### `server.py`

This module contains the implementation of a Python server that can communicate with the client over UDP and TCP connections. The server receives messages from the client, converts them to uppercase letters, and it can alse sends the message back to the client.

##### Usage
```bash
python server.py -mode tcp
# or
python server.py -mode udp
```

- `socketSERVER`: An abstract base class that defines the common methods for the TCP and UDP server classes.
- `tcpSERVER`: A concrete subclass that implements the TCP server.
- `udpSERVER`: A concrete subclass that implements the UDP server.
- `start`: Starts the server and waits for incoming client connections.
- `broadcast`: Sends messages to all connected clients.
close: Closes all connected client sockets and stops the server.
- `__accept_connection`: Accepts a new client connection and starts a new thread to handle incoming messages from the client.
- `__recv`: Receives messages from the client over and prints them in uppercase letters.

#### `module.py` 

This module contains helper methods for reliable data transfer and message formatting for both the client and server over UDP. The module consists of the following methods:

- `udp_send`: Sends UDP messages and resends them until the message is received by the server. Besides, it will packs the data and sequence number for reliable data transfer.
- `udp_recv`: Receives UDP messages and sends an acknowledgment message back to the server to ensure reliable data transfer. Also, it will Unpacks the data and sequence number for reliable data transfer.
- `__sendto`: for send the message and resend if send not sucessfully
- `__udp_md5`: MD5 hash for checksum
- `__udp_hash_mes`: combine the origial mes and checksum to message