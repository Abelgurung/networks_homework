To see all the available algos:
```bash
sysctl net.ipv4.tcp_available_congestion_control
```

To change the current algo(systemwide):

```bash
sudo sysctl -w net.ipv4.tcp_congestion_control=reno
```

To change the current algo inside code for the run (didn't test these):

In C:

```C
const char *cc = "algo";
setsockopt(sockfd, IPPROTO_TCP, TCP_CONGESTION, cc, strlen(cc));
```

In python:

```python
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b"algo")
```

Prereq:

```bash
sudo apt-get install -y build-essential linux-headers-$(uname -r)
```

Load it:

```bash
sudo insmod algo.ko
```

Check it:

```bash
sysctl net.ipv4.tcp_available_congestion_control
```

Unload it: 

```bash
sudo rmmod algo
```

Full run:

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