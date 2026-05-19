#!/usr/bin/env python3
"""
This is a utility script for documentation. It reads the agent definition
and updates the Mermaid diagram in README.md.
"""
import os
import sys
import re

# Ensure the parent directory is in sys.path so we can import cda_dust_agent
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from cda_dust_agent.agent import root_agent

README_PATH = os.path.join(PROJECT_ROOT, "README.md")

def generate_mermaid_flowchart() -> str:
    """Reads the root ADK agent and builds a Mermaid diagram string."""
    names = [sub.name for sub in root_agent.sub_agents]
    
    lines = ["```mermaid", "flowchart LR"]
    lines.append("    subgraph \"CDA_Dust_Analyzer_Agent (Sequential Agent)\"")
    
    # Generate the nodes and their simple connections
    for i, name in enumerate(names):
        if i == 0:
            lines.append(f"        {name}([{name}])")
        else:
            prev_name = names[i - 1]
            lines.append(f"        {prev_name} --> {name}([{name}])")
            
    lines.append("    end")
    lines.append("```")
    
    return "\n".join(lines)

def update_readme(mermaid_text: str):
    """Finds the architecture diagram block in README.md and overwrites it."""
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found.")
        return

    with open(README_PATH, "r") as f:
        content = f.read()

    # Regex to match the existing picture or an existing mermaid block
    # Looks for a section starting with ### Architecture and trailing down to ### Key Features
    pattern = re.compile(
        r"(### Architecture\n)(.*?)(?=\n### Key Features)",
        re.DOTALL
    )

    if pattern.search(content):
        new_content = pattern.sub(f"\\1{mermaid_text}", content)
        with open(README_PATH, "w") as f:
            f.write(new_content)
        print(f"Successfully updated {README_PATH} with dynamic Mermaid diagram.")
    else:
        print("Error: Could not find '### Architecture' section in README.md.")


if __name__ == "__main__":
    mermaid_code = generate_mermaid_flowchart()
    update_readme(mermaid_code)
