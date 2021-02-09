## Meetings
-----------
Tracking for meeting date, time, duration, and content.

Creation Date: January 15, 2021

Team: The Nerd Herd

Author: Zeke Petersen


### 1/13/2021 - Initial team meeting
------------------------------------
6:00pm-6:20pm (all members present)
* Divided up the project into two phases:
  * Phase 1: Create core modules outlined in project document
  * Phase 2: Integrate modules and build tree
* Divided up initial work into the following roles:
  * Logan - Preprocessing
  * Jarett - Modeling
  * Jack - Data Visualization
  * Callista - Pipelining/ Initial tree design
  * Zeke - Team Leader
* Goal before first meeting with client (scheduled early on 1/19 at 12pm) will be to become familiar with assigned module and identify clarifying questions

### 1/19/2021 - Mini team meeting
---------------------------------
12:00pm-12:20pm (all members present)
* Established Thursdays at 6:00pm as weekly checkin meetings
* Clarified that Callista will begin creating the n-ary tree

### 1/20/2021 - First meeting with client
-----------------------------------------
6:00pm-6:15pm (all members present)
* Clarified questions about the project-- important takeaways included:
    * The end result will essentially be an API for a programmer that wishes to create and modify trees.
    * Use DFS to minimize memory for storing the tree.
    * "Backtracking" is not precisely what was meant-- that was more of another way to say tree manipulations.
    * Preprocessing will take some reasonable judgement calls, for example, when missing data can be imputed or not or how to add timestamps to begin with.

### 1/21/2021 - First mandatory meeting
---------------------------------------
10:30am-10:50am (Zeke, Logan, Callista)
* Presented documents that we have now
* Clarified that we only have one user class to consider (data scientist) and example use cases are in the Jupyter Notebooks provided
* Clarified that the diagrams are just a graphical schematic for how the modules fit together
* Clarified that we can and likely should be wrapping the pandas functions that are the same as the project specified functions
* The feedback is concern over how little code has been written thus far, however, since we have plans for self-imposed deadlines, it may be nothing to worry about

### 1/21/2021 - Weekly team meeting
-----------------------------------
6:00pm-6:30pm (all members present)
* Established next Thursday (1/28) as the deadline for finalizing the modules and basic tree operations
* Discussed overall system design

### 1/26/2021 - Mini team meeting
-----------------------------------
7:00pm-7:30pm (Zeke, Logan, Jack, Jarett)
* Talked about getting a static pipeline (with no tree involved) working as intended as the primary goal to get done by Thursday (This weekend at the latest)
* Emailed asking to meet Juan on Thursday at 5:30 to determine how matrices are supposed to be passed to the modeling functions.
  * We have working functions to create the matrices from an input file-- though Logan will modify the timestamps to reflect the relative time for the datapoint (essentially combining the date and time of day columns into one).
  * We are working off the assumption that we need to split the train data into train and validation data, which is then passed to the model to create forecast data. This would then be compared to the existing test data (or the test data we pre-split if we split into three sets rather than 2) using the mse/mape functions.
* Introduced the tracking file for hours, etc.

### 1/28/2021 - Weekly team meeting
-----------------------------------
6:00pm-6:20pm (Zeke, Logan, Jack, Callista)
* Discussed the questions we will be asking Juan tomorrow about the rigidity of the function signatures provided and to make sure our pipeline understanding is accurate.
* Clarified some code hygeine standards (headers, compartmentalizing test code, etc)

### 1/29/2021 - Meeting with client
-----------------------------------
2:00pm-2:20pm (all members present)
* Clarified that we do not have to rigidly adhere to the given function signatures
* Discussed how the design_matrix and ts2db functions work
* Discussed arguments for the mlp and rf models
* Learned that the presentation in 1 week will be a demonstration of a working or mostly working project.

### 2/1/2021 - Team meeting
---------------------------
5:30pm-6:00pm (all members present)
* Jarett and Logan will continue trying to get a pipeline sans-tree working
(essentially trying to get semi accurate output models)
* Zeke will write out the node.py classes, the tree class, and the add_node functions
* Once done, Zeke will pass off the responsibility to Jack to work on the execute
function, which will likely draw heavily from Jarett and Logan's work
* Callista will work on the saving and loading functions once Zeke has a working tree
creation script

### 2/4/2021 - Second mandatory meeting
---------------------------
10:30am-10:50am (all members present)
* Went over current progress with Juan (most of the non-tree code is working,
we have plans for dividing up the tree code we have left)
* Clarified that accuracy of our models is not important, so Jarett and Logan
can redirect their efforts
* Juan suggested that Callista use pickles for saving Python objects
* Discussed that we should briefly discuss how the system works, but focus tomorrow's
presentation efforts on team organization and a demo.

### 2/4/2021 - Weekly team meeting
----------------------------------
6:00pm-6:15pm (Zeke, Jack, Logan, Callista)
* (For tomorrow's presentation) Zeke and Logan have created a short presentation
slideshow and will talk about team organization and system structure.
Jarett will give a short demo showing what we have.
* Jack will continue working on the execute functions (tree.py)
* Callista will continue working on loading/saving with pickles (separate file for now)
* Logan will start on tree replication (separate file for now)
* Once the presentation is done, we will check in and get a time sharing system of
sorts working for who can work on tree.py since most of the remaining code will be
in that file. We will also assign out cleanup and the last function duties this weekend.

### 2/7/2021 - Weekend meeting
------------------------------
2:00pm-2:20pm (all members present)
* Zeke will adjust node and tree classes to reflect Jarett's updates to ts2db
* Zeke will pass off to Logan for the replicate and add tree/path functions
* Callista will finish loading/saving with pickles
* Once Logan is done, he will pass off to either Jarett or Jack for the execute and
or replace operator functions respectively
* Jarett will also assist in checking that work has been done right from our original modules
