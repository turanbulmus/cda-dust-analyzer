import json
import re

# 1. Read run_3p_ablation_study_dual.py
with open("scripts/run_3p_ablation_study_dual.py", "r", encoding="utf-8") as f:
    code = f.read()

# Helper to extract functions/definitions from the script
def extract_block(pattern, name="block"):
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find pattern for {name}")
    return match.group(1).strip()

prompts_code = extract_block(r'(SYSTEM_INSTRUCTION_TEXT = ""\".*?GENERAL_PROFILES = \{.*?\n\})', "prompts")
image_func_code = extract_block(r'(def generate_dual_spectrum_image_bytes\(.*?\n    return buf\.getvalue\(\)\n)', "generate_dual_spectrum_image_bytes")
preprocess_code = extract_block(r'(def download_and_preprocess_data\(.*?\n    return df\n)', "download_and_preprocess_data")
explanation_code = extract_block(r'(async def generate_explanation\(.*?\n        return None\n)', "generate_explanation")
pool_code = extract_block(r'(async def generate_explanations_pool\(.*?\n    return explanations_pool\n)', "generate_explanations_pool")
batch_file_code = extract_block(r'(def create_ablation_batch_input_file\(.*?\n    return output_file\n)', "create_ablation_batch_input_file")
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
            
        # Match preprocess cell
        elif "def download_and_preprocess_data()" in cell_str:
            print("Updating preprocess cell...")
            combined = image_func_code + "\n\n" + preprocess_code + "\n\n" + "df = download_and_preprocess_data()"
            set_source_from_code(cell, combined)
            modified_count += 1
            
        # Match explanation cell
        elif "async def generate_explanation(" in cell_str:
            print("Updating generate_explanation & generate_explanations_pool cell...")
            combined = explanation_code + "\n\n" + pool_code + "\n\n" + (
                "configs = Config()\n"
                "client = genai.Client()\n"
                "model_id = configs.agent_settings.model\n"
                "explanations_pool = await generate_explanations_pool(client, model_id, train_df, train_sclks)"
            )
            set_source_from_code(cell, combined)
            modified_count += 1
            
        # Match batch input file creation cell
        elif "def create_ablation_batch_input_file(" in cell_str:
            print("Updating create_ablation_batch_input_file cell...")
            combined = batch_file_code + "\n\n" + (
                "k_values = [4]\n"
                "output_dir = \"study_results_3p_dual\"\n"
                "local_jsonl = os.path.join(output_dir, \"ablation_requests_3p_dual.jsonl\")\n"
                "create_ablation_batch_input_file(test_df, explanations_pool, k_values, local_jsonl)"
            )
            set_source_from_code(cell, combined)
            modified_count += 1
            
        # Match results parsing cell
        elif "def download_and_parse_batch_results(" in cell_str:
            print("Updating download_and_parse_batch_results cell...")
            # We keep the function but change the execution block at the end
            new_source = []
            for line in source:
                line = line.replace("results_3p_ablation.parquet", "results_3p_ablation_dual.parquet")
                new_source.append(line)
            cell["source"] = new_source
            modified_count += 1
            
        # Match metrics plot cell
        elif "def generate_metrics_and_plots(" in cell_str:
            print("Updating generate_metrics_and_plots cell...")
            combined = metrics_func_code + "\n\n" + "generate_metrics_and_plots(results_df, output_dir)"
            set_source_from_code(cell, combined)
            modified_count += 1

# 4. Save new notebook
output_nb_path = "Notebooks/08_3p_ablation_study_dual.ipynb"
with open(output_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Done! Modified {modified_count} cells. Saved new notebook to {output_nb_path}")
