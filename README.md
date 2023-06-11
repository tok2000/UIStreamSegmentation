# Segmentation of User Interaction Streams for Online Process Analytics

<sub>
written by <a href="mailto:tim.krause@students.uni-mannheim.de">Tim Oliver Krause</a><br />
</sub>

## About
This repository contains the implementation, data, evaluation scripts, and results as described in my Bachelor Thesis <i>Segmentation of User Interaction Streams for Online Process Analytics</i>, submitted on June 6, 2023, to Prof. H. van der Aa of the DWS group at the University of Mannheim.

## Setup and Usage

### Installation instructions
**The project requires python >= 3.9**

1. create a virtual environment for the project 
2. install the dependencies in requirements.txt, e.g., using pip <code> pip install -r requirements.txt </code>

### Directories
The following default directories are used for input and output.

* Input: <code>logs/uilogs</code>
* Output <code>output/</code>

## Evaluation
### Results from the paper and additional results
The results reported in the paper can be obtained using this [Python notebook](https://gitlab.uni-mannheim.de/processanalytics/task-recognition-from-event-stream/-/blob/main/evaluation_paper.ipynb). It also contains additional results that we could not include in the paper due to space reasons.
### Data
Our approach was tested on a collection of user interaction log based on real task executions.
The raw data and logs used are located in <code>logs/uilogs</code>, we used the script <code>util/data_util.py</code> to create the logs by randomly merging instances of different task types.
### Reproduce
The obtain all results run the evaluation script using <code>python benchmarking.py</code>, the <code>run</code> function accepts a parameter <code>parallel</code>, which can be set to <code>True</code> to speed things up, but this requires some resources as it runs many processes in parallel.

* L1 is based on data from  [Leno et al.](https://doi.org/10.6084/m9.figshare.12543587)
* L2 is based on data from [Leno et al.](https://doi.org/10.6084/m9.figshare.12543587) and [Agostinelli et al.](https://gitlab.uni-mannheim.de/processanalytics/task-recognition-from-event-stream/-/blob/main/logs/raw/agostinelli.xes) 
* L3 is based on data from Abb & Rehse described [here](https://link.springer.com/chapter/10.1007/978-3-031-16103-2_7).


## Misc
The completion action set and the overhead indicators are defined in [the project constants](https://gitlab.uni-mannheim.de/processanalytics/task-recognition-from-event-stream/-/blob/main/const.py)
