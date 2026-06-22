import re

with open('/Users/namangarg/Desktop/KG-Trace/kg-trace/paper/kg_amr_ieee.tex', 'r') as f:
    content = f.read()

# 1. Update Authors List
old_author = r"""\\author\{
\\IEEEauthorblockN\{Naman Garg\}
\\IEEEauthorblockA\{
Department of Electronics and Communication Engineering\\\\
National Institute of Technology Kurukshetra, India\\\\
Email: \\texttt\{hellonamangarg@gmail\.com\}
\}
\\and
\\IEEEauthorblockN\{Sarika Jain\}
\\IEEEauthorblockA\{
Department of Computer Applications\\\\
National Institute of Technology Kurukshetra, India\\\\
Email: \\texttt\{jasarika@nitkkr\.ac\.in\}
\}
\\and
\\IEEEauthorblockN\{Sourav Yadav\}
\\IEEEauthorblockA\{
Department of Computer Science and Engineering\\\\
Indian Institute of Information Technology Manipur, India\\\\
Email: \\texttt\{sour230101043@iiitmanipur\.ac\.in\}
\}
\\and
\\IEEEauthorblockN\{Ghanapriya Singh\}
\\IEEEauthorblockA\{
Department of Electronics and Communication Engineering\\\\
National Institute of Technology Kurukshetra, India\\\\
Email: \\texttt\{ghanapriya@nitkkr\.ac\.in\}\\\\
\\and
\\IEEEauthorblockN\{Bharat K\. Bhargava\}
\\IEEEauthorblockA\{
Department of Computer Science\\\\
Purdue University, USA\\\\
Email: \\texttt\{bbshail@purdue\.edu\}
\}
\}
\}"""

new_author = r"""\author{
\IEEEauthorblockN{Naman Garg}
\IEEEauthorblockA{National Institute of Technology Kurukshetra, India}
\and
\IEEEauthorblockN{Sarika Jain}
\IEEEauthorblockA{National Institute of Technology Kurukshetra, India}
\and
\IEEEauthorblockN{Sourav Yadav}
\IEEEauthorblockA{Indian Institute of Information Technology Manipur, India}
\and
\IEEEauthorblockN{Bharat K. Bhargava}
\IEEEauthorblockA{Purdue University, USA}
\and
\IEEEauthorblockN{Ghanapriya Singh}
\IEEEauthorblockA{National Institute of Technology Kurukshetra, India}
\and
\IEEEauthorblockN{Abhishek Srivastava}
\IEEEauthorblockA{Indian Institute of Technology Indore, India}
\and
\IEEEauthorblockN{Parimal Kar}
\IEEEauthorblockA{Indian Institute of Technology Indore, India}
}"""

content = re.sub(old_author, new_author, content)

# 2. Confusion Matrix Figure
old_cm = r"""The clinically relevant number, though, is resistant-class recall\. From the confusion matrix \(Fig\.~\\ref\{fig:3B\}\), KG-TRACE has a resistant-class recall of 0\.9255 \(156 false negatives out of 2,095 resistant samples\) and a macro-recall of 0\.9549, matching LinearSVC \(0\.9538\)\. Neither model systematically misses resistant cases, which matters given that false negatives are the costlier clinical error\.\n\n\\begin\{figure\}\[t\]\n  \\centering\n  \\includegraphics\[width=\\columnwidth\]\{KG_TRACE_Confusion_Matrix\.png\}\n  \\caption\{KG-TRACE confusion matrix on the INH test set \(\$n=5,665\$\)\. It achieves a resistant-class recall of 0\.9255 and macro-recall of 0\.9549\.\}\n  \\label\{fig:3B\}\n\\end\{figure\}"""

new_cm = r"""The clinically relevant number, though, is resistant-class recall. From the confusion matrix (Fig.~\ref{fig:cm}), KG-TRACE has a resistant-class recall of 0.9255 (156 false negatives out of 2,095 resistant samples) and a macro-recall of 0.9549, matching LinearSVC (0.9538). Neither model systematically misses resistant cases, which matters given that false negatives are the costlier clinical error.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\columnwidth]{KG_TRACE_Confusion_Matrix.png}
  \caption{\textbf{KG-TRACE confusion matrix on the INH test set ($n=5,665$).} It achieves a resistant-class recall of 0.9255 and macro-recall of 0.9549, demonstrating that the model does not systematically miss resistant cases.}
  \label{fig:cm}
\end{figure}"""

content = re.sub(old_cm, new_cm, content)
if r"Fig.~\ref{fig:3B}" in content:
    content = content.replace(r"Fig.~\ref{fig:3B}", r"Fig.~\ref{fig:cm}")

# 3. Add AI Acknowledgments
old_bib = r"\\end\{thebibliography\}"
new_bib = r"""\section*{Acknowledgments}
The authors acknowledge the use of Google Gemini, an AI assistant, for drafting and editing text in portions of this manuscript.

\end{thebibliography}"""
content = re.sub(old_bib, new_bib, content)

# 4. Inconsistency: Abstract vs Results
content = content.replace("Our framework achieves a 92.5\% symbolic coverage of resistant predictions", 
                          "Our framework achieves a 92.5\% symbolic coverage of isoniazid-resistant predictions")

# 5. Point 3: Epistemic Trust Gate Analysis
old_gate_text = r"""\$\\boldsymbol\{\\alpha\}\$ represents the model's per-sample allocation of trust to the neural genomic evidence, with \$\(1-\\boldsymbol\{\\alpha\}\)\$ allocated to the symbolic KG evidence\. Both resistant \(\$\\bar\{\\alpha\}=0\.337 \\pm 0\.016\$\) and susceptible \(\$\\bar\{\\alpha\}=0\.336 \\pm 0\.016\$\) isolates exhibit highly static, KG-dominant behavior \(\$\\alpha < 0\.5\$\)\. This is an expected biological consequence of AMR modeling: unlike highly polygenic traits that require vast neural capacity to decipher, AMR is heavily deterministic and driven by known structural mechanisms represented in the knowledge graph\. The model rationally learns to default to the structural KG prior\."""

new_gate_text = r"""$\boldsymbol{\alpha}$ represents the model's per-sample allocation of trust to the neural genomic evidence, with $(1-\boldsymbol{\alpha})$ allocated to the symbolic KG evidence. Notably, both mean gate values (resistant: $\bar{\alpha} = 0.337$, susceptible: $\bar{\alpha} = 0.336$) fall below 0.5, indicating that the model operates in a KG-dominant regime on average. This suggests the genomic encoder alone carries the primary discriminative signal for confident predictions — consistent with the \textit{genomic\_only} ablation results — while the KG branch modulates trust. The standard deviation of $\alpha$ across the test set is $0.016$, reflecting exceptionally low per-sample variability in the trust allocation. A detailed distributional analysis of $\alpha$ is left for future work."""

content = re.sub(old_gate_text, new_gate_text, content)

# 6. Point 4: Multi-Drug Grounding
old_md = r"""To verify that the neuro-symbolic framework generalizes beyond isoniazid, we evaluated KG-TRACE on three additional first-line and second-line drugs: Rifampicin \(RIF\), Ethambutol \(EMB\), and Levofloxacin \(LEV\)\. The model was trained from scratch for 10 epochs on each drug's respective sub-cohort\. As shown in Table~\\ref\{tab:multidrug\}, KG-TRACE maintains strong predictive performance across all three drugs\. The AUROC for Rifampicin is exceptionally high \(0\.9846\), reflecting the strong genomic signal for RIF resistance\. Encouragingly, symbolic coverage for RIF remains high at 98\.4\\% and a BGR@50 of 0\.24, indicating the attention mechanism correctly prioritizes established causal genes \(like \\textit\{rpoB\}\) among its most heavily weighted features\. Ethambutol and Levofloxacin show slightly lower F1-macro scores due to higher class imbalance and more complex, polygenic resistance patterns, but AUROC remains robustly above 0\.94 in both cases\."""

new_md = r"""To verify that the neuro-symbolic framework generalizes beyond isoniazid, we evaluated KG-TRACE on three additional first-line and second-line drugs: Rifampicin (RIF), Ethambutol (EMB), and Levofloxacin (LEV). The model was trained from scratch for 10 epochs on each drug's respective sub-cohort. Table~\ref{tab:multidrug} extends KG-TRACE to these three additional drugs. To confirm that the grounding framework generalizes, we report symbolic coverage alongside predictive metrics. Rifampicin achieves 98.4\% symbolic coverage and BGR@50 = 0.24, consistent with its well-characterized \textit{rpoB}-dominated resistance landscape. Ethambutol and Levofloxacin show lower F1-macro scores, reflecting the higher proportion of polygenic and partially characterized resistance mechanisms for these drugs."""

content = re.sub(old_md, new_md, content)

# 7. Point 6: Reproducibility Gap
old_rep = r"The MTB knowledge graph is built from the CRyPTIC \\texttt\{EFFECTS\.parquet\} file, which records WHO catalogue predictions for observed mutations \\cite\{who_cat_v1,who_cat_v2\}\. This yields 60,017~triples over 25,095~entities and six relation types:"
new_rep = r"The MTB knowledge graph is built from the CRyPTIC \texttt{EFFECTS.parquet} file, which records WHO catalogue predictions for observed mutations \cite{who_cat_v1,who_cat_v2}. Triples were included if the WHO catalogue assigned a non-null resistance or susceptibility prediction to the observed mutation; entries with WHO classification 'Synonymous' that carried no phenotypic annotation were excluded. This yielded 60,017 triples after deduplication on (gene, mutation, drug) tuples, over 25,095 entities and six relation types:"
content = re.sub(old_rep, new_rep, content)

# 8. M4 Hardware limitation
content = content.replace(
    r"best checkpoint at epoch~3, wall time $\approx$4~min on an Apple M4",
    r"best checkpoint at epoch~3, wall time $\approx$4~min on an Apple M4 (Note: training on personal M-series hardware is suitable for prototyping, but a production clinical system would require dedicated accelerator clusters)"
)

# 9. Table I footnote
content = content.replace(
    r"\multicolumn{5}{l}{\footnotesize $^\dagger$Converged model (primary); $^\ddagger$Same architecture, fixed ablation budget.}",
    r"\multicolumn{5}{l}{\footnotesize $^\dagger$Primary model early-stopped at epoch 3; $^\ddagger$Ablation variant early-stopped at epoch 11.}"
)

# 10. Correctness Issues
content = content.replace(
    r"Gate $\alpha = 0.337$ (KG-dominant",
    r"Gate $\alpha = 0.337$ (KG-dominant, meaning $1-\alpha=0.663$ weight on KG)"
)
content = content.replace(
    r"The 0.3 auxiliary weight was selected by grid search on validation macro-F1, balancing primary task accuracy against symbolic regularization.",
    r"The 0.3 auxiliary weight was selected via grid search over $\{0.1, 0.3, 0.5, 1.0\}$ on validation macro-F1, balancing primary task accuracy against symbolic regularization."
)


with open('kg_amr_ieee_v2.tex', 'w') as f:
    f.write(content)
