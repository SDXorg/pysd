Tests use [unittest](https://docs.python.org/3/library/unittest.html), [pytest](https://docs.pytest.org/en/stable/), [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) and [pytest-xdist](https://docs.pytest.org/en/2.1.0/xdist.html).

To run tests:
```
make tests
```

To have coverage information:
```
make cover
```

To have information of visit lines in HTML files in `htmlcov` folder:
```
make coverhtml
```

To run in multiple CPUs (e.g: for 4 CPUs):
```
make command NUM_PROC=4
```
where `command` is any of above.

