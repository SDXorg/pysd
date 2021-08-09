
# You can set these variables from the command line.
NUM_PROC = 1


help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  tests      to run tests"
	@echo "  cover      to run tests with coverage information"
	@echo "  coverhtml  to run tests and make cover report with HTML files in cover folder"
	@echo "  clean      to remove coverage information"

clean:
	rm -rf htmlcov .coverage

tests: clean
        ifeq ($(NUM_PROC), 1)
		pytest
        else
		pytest -n $(NUM_PROC)
endif

cover: clean
        ifeq ($(NUM_PROC), 1)
	pytest --cov=pysd --cov-report term
        else
	pytest --cov=pysd --cov-report term -n $(NUM_PROC)
endif

coverhtml: clean
        ifeq ($(NUM_PROC), 1)
	pytest --cov=pysd --cov-report html --cov-report term
        else
	pytest --cov=pysd --cov-report html --cov-report term -n $(NUM_PROC)
endif

