import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("test_data.csv")

# Compute stats
mean_acc = df["accuracy"].mean()
mean_loss = df["loss"].mean()

# Generate plots
plt.figure()
plt.plot(df["time"], df["accuracy"])
plt.title("Accuracy over Time")
plt.savefig("accuracy.png")
plt.close()

plt.figure()
plt.plot(df["time"], df["loss"])
plt.title("Loss over Time")
plt.savefig("loss.png")
plt.close()

# Generate LaTeX file
tex_content = f"""
\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{float}}

\\title{{Experiment Report}}
\\date{{}}

\\begin{{document}}

\\maketitle

\\section*{{Summary}}

Mean Accuracy: {mean_acc:.3f} \\\\
Mean Loss: {mean_loss:.3f}

\\section*{{Plots}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.7\\textwidth]{{accuracy.png}}
\\caption{{Accuracy over Time}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.7\\textwidth]{{loss.png}}
\\caption{{Loss over Time}}
\\end{{figure}}

\\end{{document}}
"""

with open("report.tex", "w") as f:
    f.write(tex_content)

print("Generated report.tex")
