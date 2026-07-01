# ruff: noqa
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

import base64
import json
import os
import re
from typing import Any
import google.auth
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import Gemini
from google.adk.workflow import Workflow, START
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.genai import types

from .config import config

# Only retrieve GCP project credentials if Vertex AI is explicitly enabled
if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() in ("true", "1"):
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    except Exception:
        pass


class ExpenseData(BaseModel):
    """Expense report data extracted from the incoming email/JSON event."""

    amount: float = Field(description="Expense amount in USD")
    submitter: str = Field(description="Email of the person who submitted")
    category: str = Field(description="Expense category, e.g. travel, meals")
    description: str = Field(description="What the expense is for")
    date: str = Field(description="Date of the expense (YYYY-MM-DD)")


def luhn_checksum(card_number: str) -> bool:
    """Validate a card number using Luhn's algorithm."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(divmod(d * 2, 10))
    return checksum % 10 == 0


def detect_prompt_injection(text: str) -> bool:
    """Detect potential prompt injection attempts in the description text."""
    patterns = [
        r"ignore (?:all |previous |other )?instructions",
        r"ignore (?:all |previous |other )?rules",
        r"force (?:auto-)?approval",
        r"bypass (?:review|rules|security)",
        r"auto-approve (?:this|all|expense)",
        r"set status to approved",
        r"override (?:system|instruction|rules)",
    ]
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def parse_expense_email(node_input: Any) -> Event:
    """Parse a Pub/Sub trigger event or raw input and extract expense data.

    The details sit under a 'data' key which might be base64-encoded (Pub/Sub)
    or plain JSON (local testing).
    """
    if hasattr(node_input, "parts"):
        text = "".join(part.text for part in node_input.parts if part.text)
    elif isinstance(node_input, str):
        text = node_input
    elif isinstance(node_input, dict):
        text = json.dumps(node_input)
    else:
        text = str(node_input)

    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return Event(output={"error": f"Invalid JSON: {text[:200]}"})

    data = event.get("data", {})

    if isinstance(data, str):
        try:
            data = json.loads(base64.b64decode(data).decode("utf-8"))
        except Exception:
            return Event(
                output={"error": f"Failed to decode base64 data: {data[:200]}"}
            )

    parsed = {
        "amount": float(data.get("amount", 0)),
        "submitter": data.get("submitter", "unknown"),
        "category": data.get("category", "other"),
        "description": data.get("description", ""),
        "date": data.get("date", ""),
    }
    return Event(output=parsed)


def route_by_amount(node_input: dict, ctx: Context) -> Event:
    """Route expenses based on the configured dollar threshold.

    Under the threshold -> AUTO_APPROVE route.
    Threshold or more -> NEEDS_REVIEW route.
    """
    if "error" in node_input:
        return Event(
            route="NEEDS_REVIEW",
            output=node_input,
            state={"expense_data": node_input},
        )
    amount = node_input.get("amount", 0.0)

    return Event(
        route="AUTO_APPROVE" if amount < config.review_threshold else "NEEDS_REVIEW",
        output=node_input,
        state={"expense_data": node_input},
    )


def security_checkpoint(node_input: dict, ctx: Context) -> Event:
    """Scrub PII (SSNs and Credit Cards) and defend against prompt injections.

    If prompt injection is detected, route straight to manager review and flag it.
    Otherwise, if clean, route to LLM review agent.
    """
    if "error" in node_input:
        return Event(
            route="INJECTION_DETECTED",
            output=node_input,
            state={"expense_data": node_input},
        )
    expense = dict(node_input)
    description = expense.get("description", "")

    redacted_categories = []

    # 1. Scrub SSNs
    ssn_pattern = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
    if ssn_pattern.search(description):
        description = ssn_pattern.sub("[REDACTED SSN]", description)
        redacted_categories.append("SSN")

    # 2. Scrub Credit Cards (with Luhn check to prevent false positives)
    cc_pattern = re.compile(r"\b(?:\d[- ]*?){13,19}\b")
    matches = cc_pattern.finditer(description)
    cc_to_redact = []
    for match in matches:
        raw_match = match.group(0)
        clean_num = "".join(c for c in raw_match if c.isdigit())
        if luhn_checksum(clean_num):
            cc_to_redact.append(raw_match)

    if cc_to_redact:
        for cc in cc_to_redact:
            description = description.replace(cc, "[REDACTED CREDIT CARD]")
        redacted_categories.append("Credit Card")

    expense["description"] = description

    # Update state and the expense_data in state (so the human manager sees clean data)
    ctx.state["expense_data"] = expense
    if redacted_categories:
        ctx.state["redacted_categories"] = redacted_categories

    # 3. Detect prompt injection
    if detect_prompt_injection(description):
        log_entry = {
            "severity": "WARNING",
            "message": f"SECURITY EVENT: Prompt injection attempt detected in expense description from {expense.get('submitter')}.",
            "submitter": expense.get("submitter"),
            "category": expense.get("category"),
            "redacted_categories": redacted_categories,
        }
        print(json.dumps(log_entry), flush=True)

        ctx.state["prompt_injection_detected"] = True

        return Event(
            route="INJECTION_DETECTED",
            output=expense,
            state={
                "expense_data": expense,
                "prompt_injection_detected": True,
                "redacted_categories": redacted_categories,
            },
        )

    return Event(
        route="CLEAN",
        output=expense,
        state={
            "expense_data": expense,
            "prompt_injection_detected": False,
            "redacted_categories": redacted_categories,
        },
    )


def auto_approve(node_input: dict) -> Event:
    """Auto-approve a low-value expense and log the decision."""
    msg = f"Expense auto-approved: ${node_input['amount']:.2f} from {node_input['submitter']}"
    print(msg, flush=True)
    return Event(
        output={"status": "approved", "message": msg, **node_input},
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


def emit_expense_alert(
    submitter: str,
    amount: float,
    category: str,
    risk_summary: str,
) -> dict:
    """Emit an alert for high-value expenses requiring review."""
    log_entry = {
        "severity": "WARNING",
        "message": f"Expense review alert: ${amount:.2f} from {submitter} — {risk_summary}",
        "alert_type": "expense_review",
        "submitter": submitter,
        "amount": amount,
        "category": category,
        "risk_summary": risk_summary,
    }
    print(json.dumps(log_entry), flush=True)
    return {"status": "alert_emitted", "submitter": submitter, "amount": amount}


# The LLM Review Agent node using the configured model
review_agent = LlmAgent(
    name="review_agent",
    model=Gemini(
        model=config.model,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    mode="single_turn",
    instruction="""You are an expense review agent. You receive expense reports
of $100 or more that need review before approval.

Analyze the expense and:
1. Check for risk factors: unusual category for the amount, vague description,
   suspiciously round numbers, very high value (>$1000), or potential policy
   violations.
2. Call the `emit_expense_alert` tool with the submitter, amount, category,
   and a brief risk summary explaining why this expense needs human review.
3. Return a structured review.

Your review MUST include:
- **Amount**: The expense amount
- **Submitter**: Who submitted it
- **Category**: The expense category
- **Risk level**: low, medium, or high
- **Risk factors**: What flags you found (if any)
- **Recommendation**: approve, request-more-info, or escalate""",
    input_schema=ExpenseData,
    tools=[emit_expense_alert],
)


# HITL: Pause workflow and wait for human manager approval
def request_approval(node_input, ctx: Context):
    expense = ctx.state.get("expense_data", {})
    injection_flag = ctx.state.get("prompt_injection_detected", False)

    message = "Expense requires manager approval. Approve or reject."
    if injection_flag:
        message = "WARNING: Potential prompt injection detected. Review carefully. Approve or reject."

    yield RequestInput(
        interrupt_id="decision",
        message=message,
        payload=expense,
    )


# Process the decision made by the human manager
def process_decision(node_input, ctx: Context) -> Event:
    decision = "unknown"
    if isinstance(node_input, dict):
        decision = node_input.get("decision", "unknown")
    elif isinstance(node_input, str):
        decision = "approve" if "approve" in node_input.lower() else "reject"

    approved = decision == "approve"
    expense = ctx.state.get("expense_data", {})
    status = "approved" if approved else "rejected"

    log_entry = {
        "severity": "INFO" if approved else "WARNING",
        "message": f"Expense {status} by manager",
        "decision": status,
    }
    print(json.dumps(log_entry), flush=True)

    submitter = expense.get("submitter", "unknown")
    amount = expense.get("amount", 0.0)
    category = expense.get("category", "")
    description = expense.get("description", "")
    date = expense.get("date", "")

    msg = f"${amount:.2f} expense from {submitter} has been {status}."
    if description:
        msg += f" Description: {description} ({category}) on {date}."

    return Event(
        output={"status": status, "message": msg, **expense},
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


# Wire the workflow graph
root_agent = Workflow(
    name="expense_processor",
    edges=[
        (START, parse_expense_email, route_by_amount),
        (
            route_by_amount,
            {
                "AUTO_APPROVE": auto_approve,
                "NEEDS_REVIEW": security_checkpoint,
            },
        ),
        (
            security_checkpoint,
            {
                "CLEAN": review_agent,
                "INJECTION_DETECTED": request_approval,
            },
        ),
        (review_agent, request_approval),
        (request_approval, process_decision),
    ],
)


app = App(
    root_agent=root_agent,
    name="expense_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
