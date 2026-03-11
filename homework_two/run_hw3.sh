#!/bin/bash
set -e

for x in cubic reno algo; do
	echo "=== Collecting data for $x ==="
	sysctl -w "net.ipv4.tcp_congestion_control=$x"
	python3 tcp_stats_measure.py --auto -n 10 -t 15 -i 0.2 --cc "$x" -o "generated_data/$x.csv" --json-output "generated_data/$x.json"
done

echo ""
echo "=== Generating comparison plots ==="
python3 hw3_compare_plot.py \
	--algo generated_data/algo.json \
	--cubic generated_data/cubic.json \
	--reno generated_data/reno.json \
	-o plots
