import unittest.mock

from fastapi.testclient import TestClient

from expense_agent.fast_api_app import app


def test_pubsub_subscription_normalization() -> None:
    client = TestClient(app)

    # Mock the internal TriggerRouter._run_agent_with_retry method
    mock_run = unittest.mock.AsyncMock(return_value=[])

    with unittest.mock.patch(
        "google.adk.cli.trigger_routes.TriggerRouter._run_agent_with_retry",
        mock_run,
    ):
        response = client.post(
            "/apps/expense_agent/trigger/pubsub",
            json={
                "message": {
                    "data": "eyJhbW91bnQiOiA1MH0=",  # base64 encoded {"amount": 50}
                    "messageId": "12345",
                },
                "subscription": "projects/test-project/subscriptions/test-subscription",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"status": "success"}

        # Verify that user_id was normalized to "test-subscription"
        mock_run.assert_called_once()
        kwargs = mock_run.call_args[1]
        assert kwargs["user_id"] == "test-subscription"
