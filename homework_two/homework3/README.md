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

To test with iperf:

Run an iperf server:
```bash
iperf3 -s
```

Now in a different terminal, run

```bash
python3 iperf3_client.py 127.0.0.1 -p 5201 -t 10 --cc algo
```

Full run to compile and sanity check:

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

Full run (assuming iperf server is running):

```bash
make
sudo insmod algo.ko
python3 iperf3_client.py 127.0.0.1 -p 5201 -t 10 --cc algo
sudo rmmod algo
make clean
```