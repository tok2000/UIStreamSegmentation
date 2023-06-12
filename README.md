# Segmentation of User Interaction Streams for Online Process Analytics

<sub>
written by <a href="mailto:tim.krause@students.uni-mannheim.de">Tim Oliver Krause</a><br />
</sub>

## About
This repository contains the implementation, data, evaluation scripts, and results as described in my Bachelor Thesis <i>Segmentation of User Interaction Streams for Online Process Analytics</i>, submitted on June 6, 2023, to Prof. H. van der Aa of the DWS group at the University of Mannheim.

## Setup and Usage

### Installation instructions
**The project requires python >= 3.9**

1. Create a virtual environment for the project.
2. Install the dependencies in requirements.txt, e.g., using pip <code> pip install -r requirements.txt </code>

### Directories
The following default directories are used for input and output.

* Input: <code>bernard/datasets/</code>
* Output <code>bernard/experiment/results/</code>

## Evaluation
### Data
Our approach was tested on a variety of user interaction and customer journey logs. The raw data and logs used are located in <code>bernard/datasets/</code>.

* The directory <code>bernard/datasets/real</code> contains the real-life customer journey log by Bernard et al.
* The directory <code>bernard/datasets/synthetic</code> contains the synthetically produced logs by Bernard et al.
* The directory <code>bernard/datasets/leno</code> contains the real-life user interaction logs by Leno et al.
### Results from the thesis and additional results
The results reported in the thesis can be found in the directory <code>bernard/experiment/results/results_complete</code>. The directory <code>bernard/experiment/results/additional_results</code> also contains additional results from configurations which were not included in the thesis due to performance or space reasons. All the results are saved in Excel files which are produced by the scripts. The results can be obtained in the following way. Each Excel file represents the results of one specific configuration on each of the different datasets which are then represented by the different sheets inside of the Excel file. The sheet name hereby refers to the identifier of the dataset.
* 'reimb' refers to the real-life UI log 'Reimbursement' by Leno et al.
* 'student' refers to the real-life UI log 'StudentRecord' by Leno et al.
* 'real' refers to the real-life customer-journey log by Bernard et al.
* 'delay_d' refers to one of the synthetically produced UI logs by Bernard et al., while d refers to the chosen delay. Not every results file includes all of the delays out of the directory <code>bernard/datasets/synthetic</code>.

Each Excel sheet contains several columns.
* The columns named 'TOK_wu_x' represent the results of our approach with a warmup of x events.
* The columns named 'Bernard_wu_x' represent the results of the baseline technique of Bernard et al. in an offline setting with a warmup of x events.

For example, the column 'TOK_wu_50' in the sheet 'delay_1.0' in the Excel file 'buffer-less.xlsx' represents the results of our approach with the buffer-less configuration on the synthetic log with a delay of 1.0 and a warm-up phase of 50 events.
### Reproduce
To obtain the results from Step 1 (Choice of Semantic Factor), run the evaluation script using <code>python bernard/experiment/eval_compl_semanticfactor.py</code>. The script produces six Excel files containing the evaluation results in the output directory (<code>buffer-less_semantic_fac_x.xlsx</code>) while x is replaced by each tested factor out of the evaluation in the thesis (1, 5, 10, 20, 40, 1000).

To obtain the results from Step 2 (Choice of Configuration), run the following evaluation scripts:
* Run the script for the buffer-less configuration using <code>python bernard/experiment/eval_compl_buffer-less.py</code>. The script produces one Excel file containing the evaluation results in the output directory (<code>buffer-less.xlsx</code>).
* Run the script for the standard-deviation-based configuration using <code>python bernard/experiment/eval_compl_stdev.py</code>. The script produces one Excel file containing the evaluation results in the output directory (<code>standard_dev_based.xlsx</code>).


## Misc
The completion action set and the overhead indicators are defined in [the project constants](https://gitlab.uni-mannheim.de/processanalytics/task-recognition-from-event-stream/-/blob/main/const.py)
