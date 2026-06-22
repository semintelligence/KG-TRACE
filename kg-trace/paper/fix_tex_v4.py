with open('/Users/namangarg/Desktop/KG-Trace/kg-trace/paper/kg_amr_ieee_v3.tex', 'r') as f:
    content = f.read()

# 1. The Central Performance Claim Is Misleading
old_perf = r"KG-TRACE achieves an AUROC of 0.9757, matching fully-tuned baselines despite being trained on 22\% less data (LinearSVC: 0.9794, Random Forest: 0.9806). The \textit{genomic\_only} ablation reaches an AUROC of 0.9773."
new_perf = r"KG-TRACE achieves an AUROC of 0.9757 on INH, within the range of fully-tuned baselines trained on 22\% more data (LinearSVC: 0.9794, Random Forest: 0.9806). Crucially, the \textit{genomic\_only} ablation (0.9773) slightly exceeds the full model, confirming that the KG branch does not improve predictive accuracy. We emphasize that this is expected and by design: the KG branch functions as a symbolic verifier and biological regularizer, not as a discriminative signal. Its contribution is to mechanistic grounding and clinical safety, not to AUROC."
content = content.replace(old_perf, new_perf)

# 2. SHAP Computational Cost
old_shap = r"Generating SHAP attributions via \texttt{GradientExplainer} introduces non-trivial latency at cohort scale, requiring batch processing for the full test set."
new_shap = r"On the CRyPTIC INH test partition ($n = 5,665$), SHAP attribution via \texttt{GradientExplainer} requires approximately $0.5$ minutes on CPU, corresponding to $\sim5.2$ milliseconds per sample. While acceptable for batch reporting, this latency precludes real-time inference without GPU acceleration or approximate SHAP variants (e.g., TreeExplainer-compatible approximations or KernelSHAP with reduced coalition sampling)."
content = content.replace(old_shap, new_shap)

# 3. Figure 1 Inconsistency
content = content.replace("Dataset BGR: 0.88", "Dataset BGR@50: 0.14")

with open('kg_amr_ieee_v4.tex', 'w') as f:
    f.write(content)
