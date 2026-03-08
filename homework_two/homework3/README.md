To see all the available algos:

sysctl net.ipv4.tcp_available_congestion_control

To change the current algo(systemwide):

sudo sysctl -w net.ipv4.tcp_congestion_control=reno

To change inside code:

In C:

const char *cc = "algo";
setsockopt(sockfd, IPPROTO_TCP, TCP_CONGESTION, cc, strlen(cc));

In python:
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b"algo")

Prereq:

sudo apt-get install -y build-essential linux-headers-$(uname -r)

Load it:

sudo insmod algo.ko

Check it:

sysctl net.ipv4.tcp_available_congestion_control

Unload it: 

sudo rmmod algo

Full run:

make
sudo insmod algo.ko
sysctl net.ipv4.tcp_available_congestion_control
sudo sysctl -w net.ipv4.tcp_congestion_control=algo
sysctl net.ipv4.tcp_congestion_control
sudo sysctl -w net.ipv4.tcp_congestion_control=cubic
sudo rmmod algo
make clean