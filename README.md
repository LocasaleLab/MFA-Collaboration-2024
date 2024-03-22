# MFA-Collaboration-2024


## Synopsis

Developed for 13C metabolic flux analysis (MFA), this tool processes mass isotopomer distribution (MID) data from mass spectrometry to accurately fit MID data and deduce metabolic activities, given an isotope labeling dataset. It includes MFA scripts for collaborations of analyzing U-<sup>13</sup>C-glucose tracing data in 2024.

## Requirements

The tool is built for Python 3.8 and requires the following packages:

| Packages           | Version has been tested |
|--------------------|-------------------------|
| `numpy`            | 1.22                    |
| `scipy`            | 1.7                     |
| `matplotlib`       | 3.6                     |
| `tqdm`             | 4.64                    |
| `pandas`           | 1.5.2                   |
| `sklearn`          | 3.0                     |
| `xlsxwriter`       | 3.0                     |
| `numba` (optional) | 0.56                    |


## Model

Models utilized in this software are in `scripts/model` folder.

The basic model (`base_model`) contains the base model including glycolysis, TCA cycle, pentose phosphate pathway, one-carbon metabolism, and some amino acids synthetic pathways.

The basic model with GLC and CIT buffers (`base_model_with_glc_tca_buffer`) provides two buffers to glucose and TCA cycle, which is useful in situation that only limited labeled substrate is absorbed into metabolic pathways.

The infusion model (`infusion_model`) includes an extra reaction to mimic inner unlabeled glucose production, which is useful when glucose labeling ratio is low.

## Data

All <sup>13</sup>C-isotope labeling data are in `scripts/data` folder.

These raw data are loaded and converted to standard form for MFA.

## Algorithm and Solver

Algorithm and solver utilized in this study are located in the `scripts/src/core` folder.

The `model` and `data` folder include some class definition and corresponding processing functions. Specifically, EMU
algorithm is encoded in `model/emu_analyzer_functions.py`.

Most optimizations are based on `slsqp_solver` and `slsqp_numba_solver`. As their names indicate,
the `slsqp_numba_solver` is implemented based on `numba` package for faster execution (roughly 50% time reduction).
However, the numba version has the memory leak problem in parallelized executions in Linux system. If running for long
time (longer than 50 hours), the normal version is recommended.


## Getting started

This script could also be executed as a raw Python project. Make sure Python 3.8 and all required packages are correctly
installed. First switch to a target directory and download the source code:

```shell script
git clone https://github.com/LocasaleLab/MFA-Collaboration-2024
```

Switch to the source direct, add PYTHONPATH environment and run the `main.py`:

```shell script
cd MFA-Collaboration-2024
export PYTHONPATH=$PYTHONPATH:`pwd`
python main.py
```

You could try multiple different arguments according to help information. For example:

```shell script
python main.py flux_analysis fangchao_data_fruit_fly -t
```

This instruction means running a `flux_analysis` process of data named `MFA-Collaboration-2024` in test mode (`-t`).
Detailed argument list will be explained below.

## Arguments

Usage: `python main.py running_mode job_name`

**Positional arguments**

`running_mode`: Running mode of the script.

- `flux_analysis`: Option to start a new flux analysis process to the target job.
- `result_process`: Option to process analysis results of the target job.
- `solver_output`: Option to output detailed model, data and configurations of the target job.
- `raw_experimental_data_plotting`: Option to display the raw experimental data of
  target job.

`job_name`: Name of target job. List of available jobs are listed below.

**Optional arguments**

`-p, --parallel_num`:

Number of parallel processes. If not provided, it will be selected according to CPU cores.

`-t, --test_mode`:

Whether the code is executed in test mode, which means less sample number and shorter time (several minutes).

`job_name`:

This option will execute series of operations related to analysis to experimental data. The raw data of `flux_analysis` will be stored at `scripts/output/experimental_data_analysis`, while that of `result_process` will be output
to `common_data/raw_data/experimental_data_analysis`. All tracing experiments rely on U-<sup>13</sup>C-glucose.

**List of jobs**

| Job name in this script       | Model                                                                   | Data type                                          | Total sample size <br/>(combine biological replicates) | Optimization number of each sample | Description                                                                                                                                  |
|-------------------------------|-------------------------------------------------------------------------|----------------------------------------------------|--------------------------------------------------------|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `fangchao_data_fruit_fly`     | Infusion model (`infusion_model`)                                       | Fruit flies fed with isotope-labeled food.         | 12                                                     | 10,000                             | Analyze isotope tracing data by MFA to evaluate effects of methionine resctriction diet as well as supplementary folic acids to fruit flies. |
| `fangchao_data_cultured_cell` | Basic model with GLC and CIT buffers (`base_model_with_glc_tca_buffer`) | Cultured cells treated with isotope-labeled media. | 9                                                      | 10,000                             | Analyze isotope tracing data by MFA to evaluate same effects to cultured cells.                                                              |

## Contributors

**Shiyu Liu**

+ [http://github.com/liushiyu1994](http://github.com/liushiyu1994)

## License

This software is released under the [MIT License](LICENSE-MIT).
