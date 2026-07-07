import os
import json
import base64

def main():
    json_path = "cda_dust_agent/data/output/sample_prediction.json"
    brain_dir = "/usr/local/google/home/turanbulmus/.gemini/jetski/brain/6caed9ec-1116-4f47-9bcb-aaac12b8740a"
    images_dir = os.path.join(brain_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    md_path = os.path.join(brain_dir, "sample_prediction_walkthrough.md")
    
    print(f"Reading sample prediction from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    request = data.get("request", {})
    response = data.get("response", {})
    
    md_content = []
    md_content.append("# VLM AGENT PREDICTION CALL WALKTHROUGH\n")
    md_content.append("This walkthrough shows the exact prompt structure, instructions, few-shot examples, spectrum plots, and final JSON output for a single prediction call made by the Cassini CDA Dust Analyzer Agent.\n")
    
    # 1. System Instruction
    system_instruction = ""
    system_parts = request.get("systemInstruction", {}).get("parts", [])
    if system_parts:
        system_instruction = system_parts[0].get("text", "")
    
    md_content.append("## 1. System Instruction (Agent Persona & Rules)\n")
    md_content.append("The agent is configured with the following system instructions defining Cosmic Dust Spectrometry physical principles, classification boundaries, and mandatory disqualification rules:\n")
    md_content.append("```\n" + system_instruction.strip() + "\n```\n")
    
    # 2. Few-shot reference examples & Target Spectrum
    md_content.append("## 2. Request Payload (User Prompts & Images)\n")
    md_content.append("The prompt contains classification task instructions, followed by visual few-shot reference examples, and finally the target spectrum to classify.\n")
    
    parts = request.get("contents", [{}])[0].get("parts", [])
    image_idx = 0
    
    for part in parts:
        if "text" in part and part["text"]:
            text_val = part["text"].strip()
            # If it's a general separator or example profile
            if text_val.startswith("Example:"):
                md_content.append(f"\n### {text_val}\n")
            elif "Based on the provided examples" in text_val:
                md_content.append("\n### Task Instructions\n")
                md_content.append(text_val + "\n")
            elif "Sample ID (sclk):" in text_val:
                md_content.append("\n### Target Spectrum Details\n")
                md_content.append(f"**Metadata**: `{text_val}`\n")
            else:
                md_content.append(text_val + "\n")
        
        elif "inline_data" in part and part["inline_data"]:
            img_b64 = part["inline_data"].get("data")
            mime_type = part["inline_data"].get("mime_type", "image/png")
            if img_b64:
                ext = mime_type.split("/")[-1]
                image_idx += 1
                
                # Check if this is the last image (the target) or a few-shot image
                is_last_image = (part == [p for p in parts if "inline_data" in p and p["inline_data"]][-1])
                img_name = f"target_spectrum.{ext}" if is_last_image else f"fewshot_example_{image_idx}.{ext}"
                img_path = os.path.join(images_dir, img_name)
                
                with open(img_path, "wb") as img_f:
                    img_f.write(base64.b64decode(img_b64))
                
                caption = "Target Spectrum Plot under analysis" if is_last_image else f"Few-shot Example {image_idx} Plot"
                md_content.append(f"\n![{caption}](file://{img_path})\n")
                
    # 3. Model Output Response
    md_content.append("\n## 3. Model Output Response\n")
    md_content.append("The model analyzed the target spectrum image, mapped it against the instructions/few-shots, and returned the final class prediction in structured JSON format:\n")
    
    candidates = response.get("candidates", [])
    resp_text = ""
    if candidates:
        resp_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
    md_content.append("```json\n" + resp_text.strip() + "\n```\n")
    
    print(f"Writing markdown walkthrough to {md_path}...")
    with open(md_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(md_content))
        
    print("Markdown walkthrough created successfully!")

if __name__ == "__main__":
    main()
