import asyncio
import json
from pathlib import Path

from google.adk.runners import InMemoryRunner
from google.genai import types

from expense_agent.agent import app


async def run_case(case: dict, runner: InMemoryRunner) -> dict:
    case_id = case["eval_case_id"]
    prompt_text = case["prompt"]["parts"][0]["text"]

    session = await runner.session_service.create_session(
        app_name="expense_agent", user_id="eval-user"
    )

    events = []
    need_resume = False

    # 1. First run
    async for event in runner.run_async(
        user_id="eval-user",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=prompt_text)]
        ),
    ):
        serialized_event = event.model_dump(exclude_none=True, mode="json")
        events.append(serialized_event)

        # Check if the execution is interrupted for HITL manager approval
        if event.long_running_tool_ids and "decision" in event.long_running_tool_ids:
            need_resume = True

    # 2. Resume if HITL was triggered
    if need_resume:
        decision = "approve"
        if case_id == "prompt_injection_attack":
            decision = "reject"

        print(f"[{case_id}] Intercepted HITL. Decision: {decision}")

        async for event in runner.run_async(
            user_id="eval-user",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="decision",
                            id="decision",
                            response={"decision": decision},
                        )
                    )
                ],
            ),
        ):
            serialized_event = event.model_dump(exclude_none=True, mode="json")
            events.append(serialized_event)

    # Clean up thought_signature
    for e in events:
        content = e.get("content") or {}
        for part in content.get("parts") or []:
            part.pop("thought_signature", None)

    # Find final response text
    final_text = ""
    for e in reversed(events):
        content = e.get("content") or {}
        parts = content.get("parts") or []
        texts = [p.get("text") for p in parts if p.get("text")]
        if texts:
            final_text = "".join(texts)
            break

    # Construct EvalCase
    case_result = {
        "eval_case_id": case_id,
        "prompt": case["prompt"],
        "agent_data": {
            "turns": [{"turn_index": 0, "turn_id": "turn_0", "events": events}]
        },
    }

    if final_text:
        case_result["responses"] = [
            {"response": {"role": "model", "parts": [{"text": final_text}]}}
        ]

    return case_result


async def main() -> None:
    dataset_path = Path("tests/eval/datasets/basic-dataset.json")
    output_path = Path("artifacts/traces/generated_traces.json")

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    runner = InMemoryRunner(app=app)
    results = []

    for case in data["eval_cases"]:
        print(f"Running case: {case['eval_case_id']}...")
        result = await run_case(case, runner)
        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"eval_cases": results}, f, indent=2)

    print(f"Traces written to {output_path} successfully!")


if __name__ == "__main__":
    asyncio.run(main())
