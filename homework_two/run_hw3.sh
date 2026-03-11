#!/bin/bash
for x in cubic reno algo; do
	echo $x
	sysctl -w "net.ipv4.tcp_congestion_control=$x"
	python3 tcp_stats_measure.py --auto -n 10 -t 15 -i 0.2 --cc "$x" -o "generated_data/$x.csv" --json-output "generated_data/$x.json"
done
