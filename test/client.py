# client.py
import argparse
import socket

PORT = 29500

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host",
    default="192.168.1.1",
    help="Server hostname or IP to connect to",
)
args = parser.parse_args()
HOST = args.host

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

message = "hello"
s.sendall(message.encode())

s.close()
