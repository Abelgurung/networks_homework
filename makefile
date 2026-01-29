all: p1 p2 report

p1: q1.py
	python q1.py

p2: q2.py
	python q2.py

report: report.tex q1_rtt_vs_distance.pdf hop_vs_rtt.pdf latency_breakdown.pdf
	pdflatex report.tex

clean:
	rm -f report.tex *.pdf