PYTHON = python3
PIP = pip3
UVICORN = uvicorn
VENV = venv
REQUIREMENTS_FILE = requirements.txt
MAIN_MODULE = app.main:app

.PHONY: setup run clean format check

setup:
	@if [ ! -d $(VENV) ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi
	$(VENV)/bin/$(PIP) install -r $(REQUIREMENTS_FILE)

run:
	@if [ ! -d $(VENV) ]; then \
		$(MAKE) setup; \
	fi
	$(VENV)/bin/$(UVICORN) $(MAIN_MODULE) --reload

format:
	$(VENV)/bin/black app

check:
	$(VENV)/bin/pylint --disable=missing-docstring app

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
