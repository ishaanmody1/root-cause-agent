.PHONY: setup test lint eval eval-llm demo real-eval clean

setup:            ## Install uv (if missing) and all dependencies
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-groups

test:             ## Unit tests
	uv run pytest -q

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

demo:             ## Offline demo: misleading clog, rule-based backend, writes runs/diagnose/*.html
	uv run rootcause diagnose --backend rule --kind clog --seed 6

demo-llm:         ## Same scenario with the LLM agent (needs ANTHROPIC_API_KEY)
	uv run rootcause diagnose --kind clog --seed 6 -v

eval:             ## Offline eval: naive baseline vs rule-based tracer on 48 scenarios
	uv run rootcause eval --n 48 --backend offline

eval-llm:         ## Full eval including the LLM agent column (needs ANTHROPIC_API_KEY, ~$2-5)
	uv run rootcause eval --n 48 --backend anthropic

real-eval:        ## Injected sensor faults on the real Liveline dataset (needs data/real/*.csv and a key)
	uv run rootcause real-eval --n 20

clean:
	rm -rf runs .pytest_cache .ruff_cache
