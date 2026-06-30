import json
import re

# 1. Read run_3p_ablation_study_self_correct.py
with open("scripts/run_3p_ablation_study_self_correct.py", "r", encoding="utf-8") as f:
    code = f.read()

# Helper to extract functions/definitions from the script
def extract_block(pattern, name="block"):
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find pattern for {name}")
    return match.group(1).strip()

prompts_code = extract_block(r'(SYSTEM_INSTRUCTION_TEXT = ""\".*?GENERAL_PROFILES = \{.*?\n\})', "prompts")
batch_file_code = extract_block(r'(def create_ablation_batch_input_file\(.*?\n    return output_file\n)', "create_ablation_batch_input_file")
self_correct_file_code = extract_block(r'(def create_self_correction_batch_input_file\(.*?\n    return output_file\n)', "create_self_correction_batch_input_file")
submit_job_code = extract_block(r'(def submit_and_poll_batch_job\(.*?\n        time\.sleep\(30\)\n)', "submit_and_poll_batch_job")
download_parse_code = extract_block(r'(def download_and_parse_batch_results\(.*?\n    return merged_df\n)', "download_and_parse_batch_results")
metrics_func_code = extract_block(r'(def generate_metrics_and_plots\(.*?\n    print\(f"\\nSaved metrics summary JSON to \{metrics_json_path\}"\)\n)', "generate_metrics_and_plots")

# 2. Read original notebook
notebook_path = "Notebooks/07_3p_ablation_study.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Helper to set lines from raw python string
def set_source_from_code(cell, raw_code):
    cell["source"] = [line + "\n" for line in raw_code.splitlines()]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

# 3. Modify cells
modified_count = 0
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if not source:
            continue
        cell_str = "".join(source)
        
        # Match prompts cell
        if "SYSTEM_INSTRUCTION_TEXT =" in cell_str:
            print("Updating prompts cell...")
            set_source_from_code(cell, prompts_code)
            modified_count += 1
            
        # Match batch input file creation cell
        elif "def create_ablation_batch_input_file(" in cell_str:
            print("Updating batch input file cells...")
            combined = batch_file_code + "\n\n" + self_correct_file_code + "\n\n" + (
                "k_values = [4]\n"
                "output_dir = \"study_results_3p_self_correct\"\n"
                "local_jsonl = os.path.join(output_dir, \"ablation_requests_3p.jsonl\")\n"
                "create_ablation_batch_input_file(test_df, explanations_pool, k_values, local_jsonl)"
            )
            set_source_from_code(cell, combined)
            modified_count += 1
            
        # Match batch prediction submit cell
        elif "def submit_and_poll_batch_job(" in cell_str:
            print("Updating batch predictions job submission cell...")
            combined = submit_job_code + "\n\n" + (
                "# --- STAGE 1: Multiclass Batch Prediction ---\n"
                "print(\"\\n--- STAGE 1: Multiclass Batch Prediction ---\")\n"
                "job_status = submit_and_poll_batch_job(configs, local_jsonl)"
            )
            set_source_from_code(cell, combined)
            modified_count += 1
            
        # Match download and parse batch prediction cell
        elif "def download_and_parse_batch_results(" in cell_str:
            print("Updating download_and_parse_batch_results & VLM self-correction cell...")
            combined = download_parse_code + "\n\n" + (
                "# 6. Download and Parse Stage 1 predictions\n"
                "output_parquet_stage1 = os.path.join(output_dir, \"results_3p_ablation_stage1.parquet\")\n"
                "stage1_df = download_and_parse_batch_results(configs, job_status, test_df, output_parquet_stage1)\n\n"
                "# 7. Identify samples predicted as \"3\" or \"3-P\" for Stage 2 Self-Correction\n"
                "reclass_candidates = stage1_df[stage1_df['predicted_label'].isin([\"3\", \"3-P\"])].copy()\n"
                "print(f\"\\nFound {len(reclass_candidates)} candidates predicted as '3' or '3-P' for Stage 2 self-correction.\")\n\n"
                "if len(reclass_candidates) > 0:\n"
                "    # Get corresponding rows with full spectrum data from test_df\n"
                "    reclass_test_df = test_df[test_df['sclk'].astype(str).isin(reclass_candidates['sclk'].astype(str))].copy()\n\n"
                "    # Merge Stage 1 predicted_label and explanation columns into reclass_test_df\n"
                "    candidate_info = reclass_candidates[['sclk', 'predicted_label', 'explanation']].copy()\n"
                "    candidate_info['sclk'] = candidate_info['sclk'].astype(str)\n"
                "    reclass_test_df['sclk'] = reclass_test_df['sclk'].astype(str)\n"
                "    reclass_test_df = pd.merge(reclass_test_df, candidate_info, on='sclk', how='inner')\n\n"
                "    # Generate Stage 2 Requests JSONL\n"
                "    local_jsonl_reclass = os.path.join(output_dir, \"reclassify_requests_3p.jsonl\")\n"
                "    create_self_correction_batch_input_file(reclass_test_df, explanations_pool, local_jsonl_reclass)\n\n"
                "    # Submit and poll Stage 2 job\n"
                "    print(\"\\n--- STAGE 2: VLM Self-Correction Prediction ---\")\n"
                "    job_status_reclass = submit_and_poll_batch_job(configs, local_jsonl_reclass)\n\n"
                "    # Download and Parse Stage 2 results\n"
                "    output_parquet_stage2 = os.path.join(output_dir, \"results_3p_ablation_stage2.parquet\")\n"
                "    stage2_df = download_and_parse_batch_results(configs, job_status_reclass, reclass_test_df, output_parquet_stage2)\n\n"
                "    # Merge refinement results back into predictions\n"
                "    print(\"\\nMerging refinement predictions...\")\n"
                "    refinement_map = dict(zip(stage2_df['sclk'].astype(str), stage2_df['predicted_label']))\n\n"
                "    final_preds = []\n"
                "    for _, row in stage1_df.iterrows():\n"
                "        sclk_str = str(row['sclk'])\n"
                "        pred = row['predicted_label']\n"
                "        if sclk_str in refinement_map:\n"
                "            refined_pred = refinement_map[sclk_str]\n"
                "            print(f\"  SCLK {sclk_str}: Refined prediction from {pred} -> {refined_pred}\")\n"
                "            pred = refined_pred\n"
                "        final_preds.append(pred)\n\n"
                "    stage1_df['predicted_label'] = final_preds\n\n"
                "# Save Final Integrated Results\n"
                "output_parquet = os.path.join(output_dir, \"results_3p_ablation.parquet\")\n"
                "stage1_df.to_parquet(output_parquet)\n"
                "print(f\"Saved integrated hierarchical classification results to {output_parquet}\")\n"
                "results_df = stage1_df"
            )
            set_source_from_code(cell, combined)
            modified_count += 1
            
        # Match metrics plot cell
        elif "def generate_metrics_and_plots(" in cell_str:
            print("Updating generate_metrics_and_plots cell...")
            combined = metrics_func_code + "\n\n" + "generate_metrics_and_plots(results_df, output_dir)"
            set_source_from_code(cell, combined)
            modified_count += 1

# 4. Save new notebook
output_nb_path = "Notebooks/10_3p_ablation_study_self_correct.ipynb"
with open(output_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Done! Modified {modified_count} cells. Saved new notebook to {output_nb_path}")
