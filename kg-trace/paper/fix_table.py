with open('/Users/namangarg/Desktop/KG-Trace/kg-trace/paper/kg_amr_ieee.tex', 'r') as f:
    c = f.read()

old_table = r"""\begin{tabular}{lcccc}
\toprule
\textbf{Drug} & \textbf{Total Samples} & \textbf{Resistance (\%)} & \textbf{AUROC} & \textbf{F1-macro} \\
\midrule
Rifampicin (RIF) & 37,831 & 29.1\% & \textbf{0.9853} & \textbf{0.9668} \\
Ethambutol (EMB) & 35,487 & 14.5\% & \textbf{0.9574} & \textbf{0.8668} \\
Levofloxacin (LEV) & 15,113 & 16.0\% & \textbf{0.9411} & \textbf{0.9032} \\
\bottomrule
\end{tabular}"""

new_table = r"""\begin{tabular}{lcccccc}
\toprule
\textbf{Drug} & \textbf{Total Samples} & \textbf{Resistant (\%)} & \textbf{AUROC} & \textbf{F1-macro} & \textbf{Coverage (\%)} & \textbf{BGR@50} \\
\midrule
Rifampicin (RIF) & 37,831 & 29.1\% & \textbf{0.9846} & \textbf{0.9643} & 98.4\% & 0.24 \\
Ethambutol (EMB) & 35,487 & 14.5\% & \textbf{0.9574} & \textbf{0.8668} & - & - \\
Levofloxacin (LEV) & 15,113 & 16.0\% & \textbf{0.9411} & \textbf{0.9032} & - & - \\
\bottomrule
\end{tabular}"""

c = c.replace(old_table, new_table)

with open('/Users/namangarg/Desktop/KG-Trace/kg-trace/paper/kg_amr_ieee.tex', 'w') as f:
    f.write(c)

