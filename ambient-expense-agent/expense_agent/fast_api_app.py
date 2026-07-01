# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os

import google.auth
from fastapi import FastAPI, Request
from google.adk.cli.fast_api import get_fast_api_app
from starlette.middleware.base import BaseHTTPMiddleware

from expense_agent.app_utils.telemetry import setup_telemetry
from expense_agent.app_utils.typing import Feedback

# Setup telemetry environment
setup_telemetry()

# Configure standard Python logging for console logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    _, project_id = google.auth.default()
except Exception:
    project_id = None

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

# Build the FastAPI app as an ambient service driven by Pub/Sub events
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=False,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=False,
    trigger_sources=["pubsub"],
)
app.title = "ambient-expense-agent"
app.description = "API for interacting with the Agent ambient-expense-agent"


# Middleware to normalize fully-qualified subscription paths down to short names
class NormalizePubSubBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and "trigger/pubsub" in request.url.path:
            body = await request.body()
            if body:
                try:
                    data = json.loads(body)
                    if "subscription" in data and isinstance(data["subscription"], str):
                        sub_path = data["subscription"]
                        if "/" in sub_path:
                            # Extract the short name (last segment after '/')
                            data["subscription"] = sub_path.split("/")[-1]
                        # Re-encode body and replace the receive channel and cached body
                        new_body = json.dumps(data).encode("utf-8")
                        request._body = new_body

                        async def receive():
                            return {
                                "type": "http.request",
                                "body": new_body,
                                "more_body": False,
                            }

                        request._receive = receive
                except Exception as e:
                    logger.error(f"Error normalizing Pub/Sub subscription body: {e}")
        return await call_next(request)


app.add_middleware(NormalizePubSubBodyMiddleware)


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.info(f"Feedback received: {feedback.model_dump()}")
    return {"status": "success"}


# Main execution serving on port 8080
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
