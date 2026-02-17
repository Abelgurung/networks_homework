tex_content = f"""
\\documentclass[11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{hyperref}}
\\usepackage{{graphicx}}
\\usepackage{{listings}}
\\usepackage{{booktabs}}

% --- Assignment Metadata ---
\\title{{Assignment 1: Network Latencies, Ping \\& Traceroute}}
\\author{{CS 536: Data Communication and Computer Networks, Spring 2026}}
\\date{{Due Date: January 29, 2026 @ 11:45 PM Eastern Time}}

\\begin{{document}}

\\begin{{center}}
    {{\\Large \\textbf{{Assignment 1: Network Latencies, Ping \\& Traceroute}}}} \\\\
    \\vspace{{5pt}}
    CS 536: Data Communication and Computer Networks, Spring 2026 \\\\
    Group members: Luke Luschwitz, Abel Gurung, Rob, Rachit, Zhizhen Yuan \\\\
    GitHub: \\url{{https://github.com/Abelgurung/networks_homework.git}}
\\end{{center}}

\\hrule
\\vspace{{15pt}}

% --- Section 1 ---
\\section{{Ping Test and Round-Trip Time (RTT) (50 points) }}


[Scatter Plot]


\\textbf{{Analysis of Network State}}



\\vspace{{20pt}}
\\hrule
\\vspace{{15pt}}

% --- Section 2 ---
\\section{{Latency Breakdown (50 points)}}

\\begin{{figure}}[h]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{latency_breakdown.pdf}}
    \\caption{{Breakdown of latency across different network hops.}}
    \\label{{fig:latency_breakdown}}
\\end{{figure}}

\\begin{{figure}}[h]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{hop_vs_rtt.pdf}}
    \\caption{{Correlation between the number of hops and total Round-Trip Time.}}
    \\label{{fig:hop_vs_rtt}}
\\end{{figure}}

\\textbf{{Observations and Analysis}}


\\end{{document}}
"""

with open("report.tex", "w") as f:
    f.write(tex_content)