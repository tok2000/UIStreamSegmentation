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
The results reported in the thesis can be found in the directory <code>bernard/experiment/results/results_complete/</code>. Each evaluation step has its own directory inside the <code>results_complete/</code> directory. The directory <code>bernard/experiment/results/additional_results</code> also contains additional results from configurations which were not included in the thesis due to performance or space reasons. All the results are saved in Excel files which are produced by the scripts. The results can be obtained in the following way. Each Excel file represents the results of one specific configuration on each of the different datasets which are then represented by the different sheets inside of the Excel file. The sheet name hereby refers to the identifier of the dataset.
* 'reimb' refers to the real-life UI log 'Reimbursement' by Leno et al.
* 'student' refers to the real-life UI log 'StudentRecord' by Leno et al.
* 'real' refers to the real-life customer-journey log by Bernard et al.
* 'delay_d' refers to one of the synthetically produced UI logs by Bernard et al., while d refers to the chosen delay. Not every results file includes all of the delays out of the directory <code>bernard/datasets/synthetic</code>.

### Reproduce
To obtain all results from the thesis, run the evaluation script using <code>python bernard/experiment/eval_complete.py</code>. The script produces several Excel files containing the evaluation results reported in the thesis. On top of that, for every different factor or configuration, the output file is saved as an csv file as described in the thesis. The column 'TOK_MPTAP_is_cut' hereby indicates the case ending events as described in the thesis. The resulting files are then saved as follows:
* The directory <code>bernard/experiment/results/results_complete/semantic_factor</code> contains six Excel files for the evaluation of the semantic factor which are named <code>buffer-less_semantic_fac_x.xlsx</code> while x is replaced by each tested factor out of the evaluation in the thesis (1, 5, 10, 20, 40, 1000).
* The directory <code>bernard/experiment/results/results_complete/configuration</code> contains three Excel files for the evaluation of the chosen configuration. <code>standard_dev_based.xlsx</code> contains the results of our approach with the standard-deviation-based configuration. <code>buffer-based.xlsx</code> contains the results of our approach with the buffer-based configuration. <code>percentile_0.95.xlsx</code> contains the results of our approach with the 0.95-percentile-based configuration. Additionally, the results of our approach with the buffer-less configuration are saved as <code>bernard/experiment/results/results_complete/buffer-less.xlsx</code> and each of the columns represents the results with a different number of warm-up events as discussed in the thesis in Step 3 of the evaluation (Impact of Warm-Up Phase).
* The results for the baseline comparison are saved into <code>bernard/experiment/results/results_complete/baseline_comp.xlsx</code>.
* The directory <code>bernard/experiment/results/results_complete/different_k</code> contains four Excel files for the evaluation of the hyper-parameter K for the evaluation of Beranard et al. which are named <code>bernard_k_x.xlsx</code> while x is replaced by each tested parameter K out of the evaluation in the thesis (20, 75, 100, 200).

To only obtain the results from Step 1 (Choice of Semantic Factor), run the evaluation script using <code>python bernard/experiment/eval_compl_semanticfactor.py</code>. The results are then saved into Excel files as already mentioned above.

To obtain the results from Step 2 (Choice of Configuration), run the evaluation script using <code>python bernard/experiment/eval_compl_config.py</code>. The results are then saved into Excel files as already mentioned above.

## Misc
The completion action set and the overhead indicators are defined in [the project constants](https://gitlab.uni-mannheim.de/processanalytics/task-recognition-from-event-stream/-/blob/main/const.py)
