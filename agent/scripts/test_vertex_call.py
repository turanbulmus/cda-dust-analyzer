import os
import json
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"

from google import genai
from google.genai import types

def main():
    project = "turan-genai-bb"
    print(f"Testing genai.Client with project={project}...")
    try:
        client = genai.Client(vertexai=True, project=project, location="us-central1")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello! Tell me in one sentence if you are responding using Vertex AI."
        )
        print("Success!")
        print("Response text:", response.text.strip())
    except Exception as e:
        print("Failed to call Vertex AI:", e)

if __name__ == "__main__":
    main()
