import os
from pydantic_settings import BaseSettings

class AgentSettings(BaseSettings):
    name: str = "CDA_Dust_Analyzer_Agent"
    model: str = "gemini-3.5-flash"
    project_id: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "turan-genai-bb")
    location: str = "us-central1"
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME", "turansgenaibb")
    
    # Workflow Execution Parameters
    inference_path: str = "batch" # 'local' or 'batch'
    fetch_data: bool = True
    force_new_annotations: bool = False
    test_mode: bool = False
    few_shot_n: int = 16
    test_n: int = 2000
    target_dataset: str = "L"
    few_shot_split_rules: int | dict = 3
    prompt_optimization: bool = False
    prompt_opt_iterations: int = 2

_shared_settings = AgentSettings()

class Config:
    def __init__(self):
        self.agent_settings = _shared_settings
