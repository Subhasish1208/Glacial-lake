import os
import json

def generate_benchmark_tables():
    json_path = r"c:\Users\sm080\Downloads\glacial lake dataset\dbcnet_glacial_lakes\ablation_results.json"
    if not os.path.exists(json_path):
        print("No ablation_results.json file found yet.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Order of display
    exp_keys = list(data.keys())
    
    print("\n" + "="*80)
    print("           WS-DBNet PHASE-BY-PHASE RESEARCH ABLATION SUMMARY")
    print("="*80)
    
    # Markdown Table
    md_lines = []
    md_lines.append("| Experiment / Combination | P1 (Spatial) | P2 (Context) | P3 (Fusion) | P4 (Decoder) | Loss Function | Precision (%) | Recall (%) | F1-Score (%) | mIoU (%) |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for key in exp_keys:
        item = data[key]
        loss_str = "BCE+Dice"
        if item["use_boundary"] and item["use_cldice"]:
            loss_str += "+Bound+clDice"
        elif item["use_boundary"]:
            loss_str += "+Bound"
        elif item["use_cldice"]:
            loss_str += "+clDice"

        row = f"| **{item['exp_name']}** | {item['phase1']} | {item['phase2']} | {item['phase3']} | {item['phase4']} | {loss_str} | {item['test_precision']:.2f} | {item['test_recall']:.2f} | {item['test_f1']:.2f} | **{item['test_miou']:.2f}** |"
        md_lines.append(row)

    md_table = "\n".join(md_lines)
    print(md_table)

    # LaTeX Table Generation for Research Paper
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Phase-by-phase ablation study of WS-DBNet architectural components and loss functions on the Glacial Lake Segmentation test split.}")
    latex_lines.append("\\label{tab:ablation_results}")
    latex_lines.append("\\begin{tabular}{l c c c c c c c c c}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Model / Experiment} & \\textbf{Spatial} & \\textbf{Context} & \\textbf{Fusion} & \\textbf{Decoder} & \\textbf{Loss} & \\textbf{Prec (\\%)} & \\textbf{Rec (\\%)} & \\textbf{F1 (\\%)} & \\textbf{mIoU (\\%)} \\\\")
    latex_lines.append("\\hline")

    for key in exp_keys:
        item = data[key]
        loss_str = "BCE+Dice"
        if item["use_boundary"] and item["use_cldice"]:
            loss_str += "+Bound+clDice"
        elif item["use_boundary"]:
            loss_str += "+Bound"
        elif item["use_cldice"]:
            loss_str += "+clDice"
            
        row = f"{item['exp_name'].replace('_', '\\_')} & {item['phase1']} & {item['phase2']} & {item['phase3']} & {item['phase4']} & {loss_str} & {item['test_precision']:.2f} & {item['test_recall']:.2f} & {item['test_f1']:.2f} & \\textbf{{{item['test_miou']:.2f}}} \\\\"
        latex_lines.append(row)

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    latex_table = "\n".join(latex_lines)

    # Write tables to summary file
    summary_txt_path = r"c:\Users\sm080\Downloads\glacial lake dataset\dbcnet_glacial_lakes\research_ablation_summary.txt"
    with open(summary_txt_path, 'w') as f:
        f.write("=== MARKDOWN ABLATION TABLE ===\n\n")
        f.write(md_table)
        f.write("\n\n=== LATEX ABLATION TABLE FOR PAPER ===\n\n")
        f.write(latex_table)

    print(f"\nSummary successfully saved to: {summary_txt_path}")

if __name__ == "__main__":
    generate_benchmark_tables()
