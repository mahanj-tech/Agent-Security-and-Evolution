.PHONY: install playground test lint generate-traces grade

install:
	agents-cli install

playground:
	agents-cli playground

lint:
	agents-cli lint

test:
	uv run pytest

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml
