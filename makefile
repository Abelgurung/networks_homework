all: p1 p2 report

p1: q1.py data.txt
	python3 q1.py data.txt

p2: q2.py list.txt
	python3 q2.py list.txt

report: report.tex q1_rtt_vs_distance.pdf hop_vs_rtt.pdf latency_breakdown.pdf
	pdflatex report.tex

clean:
	rm -f report.tex *.pdf