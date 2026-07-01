import unittest.mock

import pytest
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from expense_agent.agent import app


@pytest.mark.asyncio
async def test_workflow_auto_approve() -> None:
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="expense_agent", user_id="cli-user"
    )

    # 1. Under $100 should be auto-approved instantly
    # Raw message has 'data' key with base64 encoded data
    raw_message = '{"data": "eyJhbW91bnQiOiA1MC4wLCAic3VibWl0dGVyIjogImFsaWNlQGV4YW1wbGUuY29tIiwgImNhdGVnb3J5IjogInRyYXZlbCIsICJkZXNjcmlwdGlvbiI6ICJVYmVyIHJpZGUiLCAiZGF0ZSI6ICIyMDI2LTA2LTIyIn0="}'

    events = []
    async for event in runner.run_async(
        user_id="cli-user",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=raw_message)]
        ),
    ):
        events.append(event)

    has_approval = False
    for e in events:
        if e.output and isinstance(e.output, dict):
            if e.output.get("status") == "approved" and "auto-approved" in e.output.get(
                "message", ""
            ):
                has_approval = True

    assert has_approval, f"Low-value expense was not auto-approved. Events: {events}"


@pytest.mark.asyncio
async def test_workflow_hitl_review() -> None:
    # Mock LLM review agent response to avoid calling real Vertex AI
    async def mock_run_async_impl(self, ctx):  # type: ignore[no-untyped-def]
        yield Event(
            output="Mock review: low risk. Recommendation: approve.",
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Mock review: low risk. Recommendation: approve."
                    )
                ],
            ),
        )

    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="expense_agent", user_id="cli-user"
    )

    # High-value expense ($100 or more)
    raw_message = '{"data": "eyJhbW91bnQiOiAxNTAuMCwgInN1Ym1pdHRlciI6ICJib2JAZXhhbXBsZS5jb20iLCAiY2F0ZWdvcnkiOiAibWVhbHMiLCAiZGVzY3JpcHRpb24iOiAiQnVzaW5lc3MgZGlubmVyIiwgImRhdGUiOiAiMjAyNi0wNi0yMiJ9"}'

    with unittest.mock.patch(
        "google.adk.agents.llm_agent.LlmAgent._run_async_impl", mock_run_async_impl
    ):
        # 1. Start execution - it should yield RequestInput for manager decision
        events = []
        async for event in runner.run_async(
            user_id="cli-user",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text=raw_message)]
            ),
        ):
            events.append(event)

        # 2. Resume session with approval decision
        resume_events = []
        async for event in runner.run_async(
            user_id="cli-user",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="decision",
                            id="decision",
                            response={"decision": "approve"},
                        )
                    )
                ],
            ),
        ):
            resume_events.append(event)

        has_approved_status = False
        for e in resume_events:
            if e.output and isinstance(e.output, dict):
                if e.output.get(
                    "status"
                ) == "approved" and "has been approved" in e.output.get("message", ""):
                    has_approved_status = True

        assert has_approved_status, (
            f"Workflow did not resume and approve. Events: {resume_events}"
        )
