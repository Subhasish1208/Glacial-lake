import os
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_benchmark_tables():
    json_path = r"c:\Users\sm080\Downloads\glacial lake dataset\dbcnet_glacial_lakes\ablation_results.json"
    if not os.path.exists(json_path):
        print("No ablation_results.json file found yet.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    exp_keys = list(data.keys())

    features_def = [
        ("Strip Conv (1a)", lambda d: d["phase1"] in ["1a", "1ab"]),
        ("Hybrid Path (1b)", lambda d: d["phase1"] in ["1b", "1ab"]),
        ("Wavelet Scan (2a)", lambda d: d["phase2"] in ["2a", "2ab"]),
        ("Wavelet Patch (2b)", lambda d: d["phase2"] in ["2b", "2ab"]),
        ("ECA-FFM (3a)", lambda d: d["phase3"] in ["3a", "3ab"]),
        ("Spatial Gate (3b)", lambda d: d["phase3"] in ["3b", "3ab"]),
        ("Prog CMM (4)", lambda d: d["phase4"] != "v1"),
        ("clDice Loss", lambda d: d.get("use_cldice", False)),
        ("Boundary Loss", lambda d: d.get("use_boundary", False)),
    ]

    feature_names = [f[0] for f in features_def]

    print("\n" + "="*110)
    print("           WS-DBNet FULL FEATURE MATRIX ABLATION STUDY SUMMARY (TICKS / BLANKS)")
    print("="*110)

    # 1. MARKDOWN TICK MATRIX TABLE
    md_lines = []
    md_header = "| Experiment | " + " | ".join(feature_names) + " | Precision (%) | Recall (%) | F1-Score (%) | mIoU (%) |"
    md_sep = "| :--- | " + " | ".join([":---:"] * len(feature_names)) + " | :---: | :---: | :---: | :---: |"
    md_lines.append(md_header)
    md_lines.append(md_sep)

    for key in exp_keys:
        item = data[key]
        ticks = ["✓" if f[1](item) else "-" for f in features_def]
        row = f"| **{key}** | " + " | ".join(ticks) + f" | {item['test_precision']:.2f} | {item['test_recall']:.2f} | {item['test_f1']:.2f} | **{item['test_miou']:.2f}** |"
        md_lines.append(row)

    md_table = "\n".join(md_lines)
    print(md_table)

    # 2. LATEX TICK MATRIX TABLE FOR PAPER
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Full Architectural Feature Matrix Ablation Study of WS-DBNet on the Glacial Lake Segmentation test split.}")
    latex_lines.append("\\label{tab:ablation_tick_matrix}")
    latex_lines.append("\\begin{tabular}{l " + "c "*len(features_def) + "c c c c}")
    latex_lines.append("\\hline")
    latex_header = "\\textbf{Experiment} & " + " & ".join([f"\\textbf{{{name}}}" for name in feature_names]) + " & \\textbf{Prec (\\%)} & \\textbf{Rec (\\%)} & \\textbf{F1 (\\%)} & \\textbf{mIoU (\\%)} \\\\"
    latex_lines.append(latex_header)
    latex_lines.append("\\hline")

    for key in exp_keys:
        item = data[key]
        ticks = ["\\checkmark" if f[1](item) else "-" for f in features_def]
        row = f"{key.replace('_', '\\_')} & " + " & ".join(ticks) + f" & {item['test_precision']:.2f} & {item['test_recall']:.2f} & {item['test_f1']:.2f} & \\textbf{{{item['test_miou']:.2f}}} \\\\"
        latex_lines.append(row)

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    latex_table = "\n".join(latex_lines)

    # Write tables to summary file
    summary_txt_path = r"c:\Users\sm080\Downloads\glacial lake dataset\dbcnet_glacial_lakes\research_ablation_summary.txt"
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("=== MARKDOWN FEATURE MATRIX ABLATION TABLE ===\n\n")
        f.write(md_table)
        f.write("\n\n=== LATEX FEATURE MATRIX ABLATION TABLE FOR RESEARCH PAPER ===\n\n")
        f.write(latex_table)

    print(f"\nSummary successfully saved to: {summary_txt_path}")

if __name__ == "__main__":
    generate_benchmark_tables()
