BIN = .venv/bin
PIP = $(BIN)/pip
PYTEST = $(BIN)/pytest

.PHONY: develop
develop:
	rm -rf .venv
	python -m venv .venv
	$(PIP) install -U pip
	$(PIP) install wheel
	$(PIP) install \
		-e ../model/. \
		-r requirements/dev.txt \
		-r requirements/training.txt

.PHONY: test
test:
	$(PYTEST) -s --cov=src .
