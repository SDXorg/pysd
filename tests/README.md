Test suite
==========

Preparing the test suite
------------------------

In order to run the test:

1. Clone the repo with the --recursive flag (if not done yet).
2. Potentially create or enter a virtual environment.
3. Install test dependencies using *pip install -r tests/requirements.txt* or *conda install --file tests/requirements.txt*.

Running the test suite
----------------------

To run tests:

```shell
make tests
```

To have coverage information:

```shell
make cover
```

To have information of visit lines in HTML files in `htmlcov` folder:

```shell
make coverhtml
```

To run in multiple CPUs (e.g: for 4 CPUs):

```shell
make command NUM_PROC=4
```

where `command` is any of above.

You can also run the test using `pytest`:

```shell
python -m pytest test/
```

for running individual test, filtering warnings or some other configurations check [pytest documentation](https://docs.pytest.org).
