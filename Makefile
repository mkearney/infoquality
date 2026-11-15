VENV ?= .venv
BIN = $(VENV)/bin

.PHONY: develop
develop:
	python3 -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e "model[training,dev]"
	$(BIN)/pre-commit install

.PHONY: test
test:
	$(BIN)/pytest --cov=infoquality --cov=common --cov-report=term-missing

.PHONY: lint
lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

.PHONY: format
format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

.PHONY: clean
clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .coverage
