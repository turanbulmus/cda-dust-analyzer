import os
from pydantic_settings import BaseSettings

class AgentSettings(BaseSettings):
    name: str = "CDA_Dust_Analyzer_Agent"
    model: str = "gemini-3-pro-preview"
    project_id: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
    location: str = "global"
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME", "your-bucket-name")
    
class Config:
    def __init__(self):
        self.agent_settings = AgentSettings()
