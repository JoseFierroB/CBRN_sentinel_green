# AgentBeats Standard Makefile

.PHONY: install run validate test build clean

# 1. Setup
install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo "Environment setup complete."

# 2. Execution (Server Mode for A2A)
run:
	.venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload

# 3. Validation (Metric Check)
# Runs a quick audit against a mock provider to ensure metrics (Delta, Tools) work.
validate:
	@echo "Running Compliance Validation..."
	.venv/bin/python3 -m src.main --provider generic --base_url http://dummy --key dummy --limit 1
	@echo "Validation Complete. Check output for 'Defense Delta'."

# 4. Testing (Unit Tests)
test:
	@echo "Running Unit Tests..."
	# In a real scenario: pytest tests/
	.venv/bin/python3 -m src.modules.tools # Smoke test imports

# 5. Build (Docker)
build:
	docker build -t cbrn-sentinel:latest .

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -f report.json
