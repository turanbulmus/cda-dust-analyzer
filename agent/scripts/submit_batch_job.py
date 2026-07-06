import os
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv('.env')

project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'turan-genai-bb')
bucket_name = os.environ.get('GCS_BUCKET_NAME', 'turansgenaibb').replace("gs://", "")
gcs_source = f"gs://{bucket_name}/input/cda_dust_agent/data/input/batch_requests.jsonl"
model_id = "gemini-3.5-flash"

print(f"Initializing Vertex AI for project={project_id}...")
aiplatform.init(project=project_id, location="global")

job_display_name = f"cda-2000-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
model_name = f"publishers/google/models/{model_id}"

print(f"Submitting BatchPredictionJob: {job_display_name}...")
print(f"  Source GCS: {gcs_source}")
print(f"  Destination GCS: gs://{bucket_name}/output")

try:
    job = aiplatform.BatchPredictionJob.create(
        job_display_name=job_display_name,
        model_name=model_name,
        instances_format="jsonl",
        gcs_source=gcs_source,
        predictions_format="jsonl",
        gcs_destination_prefix=f"gs://{bucket_name}/output",
    )
    print("=================================================================")
    print(f" BATCH JOB SUBMITTED SUCCESSFULLY! ")
    print(f" Job Name: {job.name}")
    print(f" Display Name: {job.display_name}")
    print(f" Resource Name: {job.resource_name}")
    print(f" State: {job.state}")
    print("=================================================================")
except Exception as e:
    print(f"Primary model submit error: {e}")
    fallback_model = "publishers/google/models/gemini-2.5-flash"
    print(f"Submitting with fallback model {fallback_model}...")
    job = aiplatform.BatchPredictionJob.create(
        job_display_name=job_display_name,
        model_name=fallback_model,
        instances_format="jsonl",
        gcs_source=gcs_source,
        predictions_format="jsonl",
        gcs_destination_prefix=f"gs://{bucket_name}/output",
    )
    print("=================================================================")
    print(f" BATCH JOB SUBMITTED SUCCESSFULLY (FALLBACK)! ")
    print(f" Job Name: {job.name}")
    print(f" Display Name: {job.display_name}")
    print(f" Resource Name: {job.resource_name}")
    print(f" State: {job.state}")
    print("=================================================================")
