import os
from pydantic_settings import BaseSettings

class AgentSettings(BaseSettings):
    name: str = "CDA_Dust_Analyzer_Agent"
    model: str = "gemini-3.1-pro-preview"
    project_id: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
    location: str = "us-central1"
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME", "your-bucket-name")
    
    # Workflow Execution Parameters
    inference_path: str = "batch" # 'local' or 'batch'
    fetch_data: bool = False
    force_new_annotations: bool = False
    test_mode: bool = True
    few_shot_n: int = 3
    test_n: int = 100
    prompt_optimization: bool = True
    prompt_opt_iterations: int = 2

class Config:
    def __init__(self):
        self.agent_settings = AgentSettings()
