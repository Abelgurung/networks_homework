
SHELL := /bin/bash

all: report

p1: q1.py data.txt
	python3 q1.py data.txt

p2: q2.py list.txt
	python3 q2.py list.txt

report: report.tex p1 p2
	pdflatex report.tex

clean:
	rm -f *.pdf *.aux *.log
