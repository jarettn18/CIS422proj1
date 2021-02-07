## Project 1: Time Series Analysis Support For Data Scientists
---------------------------------------------------------------
README containing a description of the software and directory structure,
dependencies and instructions for installation, and a short user guide.

CIS 422

Team: The Nerd Herd

Creation date: 1/15/2021


### Project Description
-----------------------
The system contained in this repository is intended to be a tool used by Data
Scientists to rapidly test Time Series pipelines. Once tested, successful
pipelines can be saved to go into production to make actionable forecasts.


### Directory Structure
-----------------------
* ./Modules - contains the primary modules, including testing and helper functions
* ./Logistics - contains internal scheduling deadlines, team meeting notes,
and internal code standards
* ./Documentation - contains the SRS and SDS
* ./TestData - contains input .csv files used to test the various modules
* ./Transformation Tree - contains node and tree class definitions and methods


### Dependencies and Instructions for Installation
--------------------------------------------------
In order to run the software, Python 3.8 or later must be installed. Installers
can be found at python.org/downloads. To check that Python successfully installed,
enter "python --version" at the command line or terminal.

Pip should come preinstalled with Python and it will be needed to install the
software dependencies. To check that Python successfully installed,
enter "pip --version" at the command line or terminal.

* pip is used to install dependencies via the command "pip install \<dependency\>"
where each of the following will be substituted for \<dependency\> one at a time
(ex: entering "pip install scikit-learn" and then "pip install pandas" at a command line or
terminal).
    * scikit-learn
    * pandas
    * numpy
    * pyjanitor
    * seaborn
    * matplotlib
    * cpickle
    * statsmodels

### User Guide
--------------
TODO


### Authors
-----------
* Logan Levitre (llevitre@uoregon.edu)
* Jarett Nishijo (jnishijo@uoregon.edu)
* Zeke Petersen (ezekielp@uoregon.edu)
* Jack Sanders  (jsander5@uoregon.edu)
* Callista West (cwest10@uoregon.edu)
