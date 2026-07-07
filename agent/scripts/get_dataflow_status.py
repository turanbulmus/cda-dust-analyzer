import os
import googleapiclient.discovery
from google.auth import default

def main():
    credentials, project = default()
    service = googleapiclient.discovery.build('dataflow', 'v1b3', credentials=credentials)
    
    project_id = "turan-genai-bb"
    location = "us-central1"
    
    print(f"Querying Dataflow jobs in project={project_id}, location={location}...")
    try:
        request = service.projects().locations().jobs().list(
            projectId=project_id,
            location=location
        )
        response = request.execute()
        
        jobs = response.get('jobs', [])
        if not jobs:
            print("No Dataflow jobs found.")
            return
            
        print(f"Found {len(jobs)} jobs:")
        for job in jobs[:5]:
            job_name = job.get('name')
            job_id = job.get('id')
            state = job.get('currentState')
            created = job.get('createTime')
            print(f" - Job Name: {job_name} | ID: {job_id} | State: {state} | Created: {created}")
            
            if state == "JOB_STATE_RUNNING" and job_name.startswith("cda-data-prep-full"):
                # Fetch recent messages
                print(f"\nRecent logs/messages for job {job_name} ({job_id}):")
                msg_request = service.projects().locations().jobs().messages().list(
                    projectId=project_id,
                    location=location,
                    jobId=job_id,
                    minimumImportance="JOB_MESSAGE_BASIC"
                )
                msg_response = msg_request.execute()
                messages = msg_response.get('jobMessages', [])
                print(f"Total messages: {len(messages)}")
                for msg in messages[-35:]: # Print last 35 messages
                    print(f"   [{msg.get('time')}] {msg.get('messageImportance')}: {msg.get('messageText')}")
            
    except Exception as e:
        print("Failed to query Dataflow jobs:", e)

if __name__ == "__main__":
    main()
