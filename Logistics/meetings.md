## Meetings
-----------
Tracking for meeting date, time, duration, and content.

### 1/13/2021 - Initial Team Meeting
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

### 1/19/2021 - Mini Team Meeting
---------------------------------
12:00pm-12:20pm (all members present)
* Established Thursdays at 6:00pm as weekly checkin meetings
* Clarified that Callista will begin creating the n-ary tree

### 1/20/2021 - First Meeting with Client
-----------------------------------------
6:00pm-6:15pm (all members present)
* Clarified questions about the project-- important takeaways included:
    * The end result will essentially be an API for a programmer that wishes to create and modify trees.
    * Use DFS to minimize memory for storing the tree.
    * "Backtracking" is not precisely what was meant-- that was more of another way to say tree manipulations.
    * Preprocessing will take some reasonable judgement calls, for example, when missing data can be imputed or not or how to add timestamps to begin with.

### 1/21/2021 - First Mandatory Meeting
---------------------------------------
10:30am-10:50am (Zeke, Logan, Callista)
* Presented documents that we have now
* Clarified that we only have one user class to consider (data scientist) and example use cases are in the Jupyter Notebooks provided
* Clarified that the diagrams are just a graphical schematic for how the modules fit together
* Clarified that we can and likely should be wrapping the pandas functions that are the same as the project specified functions
* The feedback is concern over how little code has been written thus far, however, since we have plans for self-imposed deadlines, it may be nothing to worry about

### 1/21/2021 - Weekly Team Meeting
-----------------------------------
6:00pm-6:30pm (all members present)
* Established next Thursday (1/28) as the deadline for finalizing the modules and basic tree operations
* Discussed overall system design

### 1/26/2021 - Mini team meeting
-----------------------------------
7:00pm-7:30pm (all members except Callisa present)
* Talked about getting a static pipeline (with no tree involved) working as intended as the primary goal to get done by Thursday (This weekend at the latest)
* Emailed asking to meet Juan on Thursday at 5:30 to determine how matrices are supposed to be passed to the modeling functions.
  * We have working functions to create the matrices from an input file-- though Logan will modify the timestamps to reflect the relative time for the datapoint (essentially combining the date and time of day columns into one).
  * We are working off the assumption that we need to split the train data into train and validation data, which is then passed to the model to create forecast data. This would then be compared to the existing test data (or the test data we pre-split if we split into three sets rather than 2) using the mse/mape functions.
* Introduced the tracking file for hours, etc.
