## Initial Standards
-------------------
The following information is intended to improve modularity and later integration. Documentation of code will be done by the software engineer who wrote it, but should be easily readable by other engineers. This information will be updated as time goes on.

Creation Date: January 15, 2021

Team: The Nerd Herd

Author: Zeke Petersen


### Documentation/Organization
-----------------------------
* At the beginning of a module as a header comment, the following must be included:
  * Name of the file
  * Class and date
  * Name of the team and principal programmer
  * General overview of the file's purpose

* For *every* function, both explicitly specified in the project handout and helpers, the following must be included in a block comment:
  * Description of the function - what does it do (and why if appropriate)
  * Input and Output form - for example, int list or a tuple of the form (int, string)
  * Any quirks of data expectations - for example, if it expects ints to be non-zero

* It is assumed that the functions explicitly described in the document will have the same signature as it appears in the document.
  * UPDATE on 1/29: The handful of splitting functions may deviate as needed to work with the Transformation Tree.

* The three files that will represent modules 3.1/3.2, 3.3, and 3.4 are all expected to contain only the explicitly described functions in the project description.
  * For each module, any helpers should be relegated to a single helper file for that module if possible (with a file name that reflects that purpose, for example preprocessing_helper.py helps preprocessing.py). This should help compartmentalize code and make it easier to check progress.
  * Tests for each module (every significant function should be tested) are expected to reside in a similarly named file (for example preprocessing_tests.py).

* The Transformation Tree will be another separate module, but more accessible to the user.

* If a module requires a library or package, documentation for how to install the library or package locally in the module header must be present (through pip, Anaconda, or wherever else) so the other engineers may configure their environments to match.
  * We will later collect this information into one place to describe how to run our code.

* It is assumed that we will be using Python 3.8 for the bulk of the project.

* Tasks will be assigned at full group meetings, but will be adjusted as needed as we discover the actual time it takes to complete tasks.


### Data Expectations
-------------------
* A time series will be a csv of the form (time0,magnitude0),(time1,magnitude1),...,(timeN,magnitudeN) where time and magnitude are both non-negative values.
  * There are some input files with either the time values missing or with an extra column. These will be handled by the user using the preprocessing functions as needed.
