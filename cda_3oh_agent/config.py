import os
import logging
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentModel(BaseModel):
    """Agent model settings for the binary 3-OH vs non-3-OH CDA Dust Agent."""

    name: str = Field(default="cda_3oh_agent")
    model: str = Field(default="gemini-4-argon")
    inference_path: str = Field(default="batch")  # Options: "local", "batch"
    location: str = Field(default="us-central1")
    project_id: str = Field(default="turan-genai-bb")
    bucket_name: str = Field(default="turansgenaibb")
    gcs_input_prefix: str = Field(default="input/3oh_agent")
    gcs_output_prefix: str = Field(default="output/3oh_agent")
    few_shot_n: int = Field(default=18)
    test_n: int = Field(default=-1)
    fetch_data: bool = Field(default=True)
    force_new_annotations: bool = Field(default=True)
    prompt_optimization: bool = Field(default=True)
    prompt_opt_iterations: int = Field(default=3)
    enable_stage2_adjudicator: bool = Field(default=True)
    test_mode: bool = Field(default=False)


class Config(BaseSettings):
    """Configuration settings for the CDA 3-OH Dust Classification Agent."""

    model_config = SettingsConfigDict(
        env_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../.env"
        ),
        env_prefix="GOOGLE_",
        case_sensitive=True,
        extra="ignore",
    )

    agent_settings: AgentModel = Field(default=AgentModel())
    app_name: str = "cda_3oh_classification_app"
    CLOUD_PROJECT: str = Field(default="turan-genai-bb")
    CLOUD_LOCATION: str = Field(default="us-central1")
    GENAI_USE_VERTEXAI: str = Field(default="1")
    API_KEY: str | None = Field(default="")
