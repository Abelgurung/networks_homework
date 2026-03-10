This has been tested on Linux 5.10 VM and Ubuntu 20.04(Linux 5.15).

### Prereq:
```bash
sudo apt-get install -y build-essential linux-headers-$(uname -r)
```

### To see all the available congestion control algos:
```bash
sysctl net.ipv4.tcp_available_congestion_control
```

### To change the current algo(systemwide):

```bash
sudo sysctl -w net.ipv4.tcp_congestion_control=reno
```

### To change the current algo inside code for the run:

In C:

```C
const char *cc = "algo";
setsockopt(sockfd, IPPROTO_TCP, TCP_CONGESTION, cc, strlen(cc));
```

In python:

```python
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b"algo")
```

### Compile the Kernel module:
```bash
make
```

### Load the Congestion Control Algorithm module:

```bash
sudo insmod algo.ko
```

### Check if the algorithm changed (Gives a list of all available algorithms in the Kernel, including the default CUBIC and RENO):

```bash
sysctl net.ipv4.tcp_available_congestion_control
```

### Unload your module: 

```bash
sudo rmmod algo
```

### To test Locally with iperf:

Run a local iperf server:
```bash
iperf3 -s
```

Now in a different terminal, run

```bash
python3 iperf3_client.py 127.0.0.1 -p 5201 -t 10 --cc algo
```

### Full run to compile and sanity check (or if you want to add the module systemwide):

```bash
make
sudo insmod algo.ko
sysctl net.ipv4.tcp_available_congestion_control
sudo sysctl -w net.ipv4.tcp_congestion_control=algo
sysctl net.ipv4.tcp_congestion_control
sudo sysctl -w net.ipv4.tcp_congestion_control=cubic
sudo rmmod algo
make clean
```

### Full run:

Make sure iperf server is running (if local), or modify the ip and port for an online server

```bash
make
sudo insmod algo.ko 
python3 ../iperf3_client.py 127.0.0.1 -p 5201 -t 10 --cc algo
sudo rmmod algo
make clean
```