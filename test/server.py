# server.py
import argparse
import socket

PORT = 29500

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host",
    default="192.168.1.1",
    help="Address to bind to (e.g. 0.0.0.0 for all interfaces, or 192.168.1.2 on that machine)",
)
args = parser.parse_args()
HOST = args.host

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

print(f"Listening on {HOST}:{PORT}...")
conn, addr = s.accept()
print(f"Connected by {addr}")

data = conn.recv(1024)
print("Received:", data.decode())

conn.close()
s.close()
