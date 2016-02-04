Models for testing system dynamics translation, simulation, and display software
================================================================================

This repository is a resource for testing system dynamics software and translation tools.
It provides a standard set of simple test cases in various formats, with a proposed canonical 
output for that test. 

Folders within the [Test](https://github.com/SDXorg/test-models/tree/master/tests/) directory 
contain models that exercise a minimal amount of functionality (such as lookup tables) for doing 
unit-style testing on translation and simulation pathways.

Folders within the [Samples](https://github.com/SDXorg/test-models/tree/master/samples/) 
directory contain complete models that can be used for integration tests, benchmarking, and demos.

Each model folder contains:

- a single model concept, with its canonical output (named `output.csv` or `output.tab`) containing (at least) 
the stock values over the standard timeseries in the model files
- Model files that produce said output (.mdl, .xmile, stella, pysd, etc)
- A text file entitled `README.md` containing:
 - The purpose of the test model (what functionality it executes)
 - The version of software that the canonical output was originally prepared by
 - The author of the test model and contact info
 - Submission Date
- Screenshots of model construction programs (optional)

For a demonstration, see the 
[teacup example](https://github.com/SDXorg/test-models/tree/master/benchmarks/teacup)


## Contributing:
All members of the SD community are invited to contribute to this repository. To do so, create a 
fork, add your contribution using one of the following methods, add yourself to the
[AUTHORS](AUTHORS) file, then submit a pull request. 

To request that a specific test be added, create an issue on the 
[issues](https://github.com/SDXorg/test-models/issues) page of this repository.

#### Expanding existing cases
Many of these cases have model files for some modeling formats but not others. To add a model file
in another format, check that your model output replicates the 'canonical example' to reasonable 
fidelity, preferably using identical varaibale names, and add an entry to the contributions table
in the directory's `README.md` file.

#### Adding new cases
To add a new case, in your local clone add a folder in either the `tests` or `benchmarks` directory
as appropriate, copy an example `README.md` file from another folder, and edit to suit your needs.

## Result formatting

To simplify tools and scripts around model validation, canonical
output files should be UTF8 tab-separated or comma-separated files.
Each row represents model results at a single timestep (rather than
each row representing a single variable's results for every timestep).

#### Getting model results from STELLA/iThink

The following process ensures that output files end up in the format
expected by tools that interact with this repository.

1. Open the model in STELLA or iThink
2. Run the model (choose Run from the Run menu)
3. From the Edit menu, choose Export Data
4. In the `Export Data` modal dialog, choose `One Time` as the Export Type
5. In the `Export Data Source`, make sure both the `Export all model
   variables` and `Every DT - Export every intermediate value during the
   run` are selected
6. For `Export Destination`, choose Browse and name the file
   `output.csv`, and make sure the left-most checkbox below Browse is
   selected.  You may have to create an empty file named `output.csv`
   manually beforehand in your operating system's file browser.  Ensure
   that of the two `Data` styles (columnar on the left, horizontal on the
   right) the left-most (columnar results) is selected.  This is the default.
7. Click `OK` at the bottom right to perform the export

#### Getting model results from Vensim

1. Open the model in Vensim
2. Run the model
3. From the `Model` menu, choose `Export Dataset...`
4. Choose the run you just performed (by default it is the file named
   `Current.vdf`
5. At the top of the dialog, in the text-box next to the `Export To` button,
   change the name of the export file from `Current` (or whatever your run
   name was) to `output`
6. Under `Export As` choose `CSV`
7. Under `Time Running` choose `down`
8. Click `OK at the bottom left to perform the export

## Tool testing

There are 2 scripts in the top level of this repo to aid in debugging,
[compare.py](compare.py) and [regression-test.py](regression-test.py).
`compare.py` expects paths to two CSV or TSV files, and will compare the
results of the two files, with some amount of smartness/fuzziness
around floating point comparisions.

`regression-test.py` can be used to compare a specific modeling tool's
output against the accepted, canonical output for a given model (which
is stored in `output.csv` in all the subdirectories of this
repository).  It can be run with external tools with the current
working directory as the root of this project:

    $ ./regression-test.py ~/src/libsd/mdl .
    $ ./regression-test.py ~/src/sd.js/mdl.js .

And it can also be run from outside of this project, for example when
this `test-models` repo is included as a [git
submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) in
another project:

    $ test/test-models/regression-test.py ./mdl test/test-models

The main requirement is that the given command
([`mdl`](https://github.com/sdlabs/libsd/blob/master/mdl.c) and
[`mdl.js`](https://github.com/sdlabs/sd.js/blob/master/mdl.js) above)
accept the path to a model as an argument, and output model results to
`stdout` in either TSV or CSV format.  If your tool requires
additional commandline args, you can specify them with quoting:

    $ ./regression-test.py "~/path/to/tool --arg1 --arg2" .

And if you have a tool that simulates Vensim models or Stella v10
models rather than xmile, you can change the model-file suffix:

    # test Vensim model files
    $ ./regression-test.py --ext mdl ~/path/to/tool .

    # test Stella v10 XMILE-variant model files
    $ ./regression-test.py --ext stmx ~/path/to/tool .

# List of Tests to add
1. Models with no stocks
2. Models with subscripted flow definitions
3. Models with different sets of subscripts on different pieces
4. Subscripted lookups
5. Test 'pi' and other keywords