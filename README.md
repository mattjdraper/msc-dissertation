# s2593817 Masters Dissertation - README
MSc Dissertation Project: "Optimizing Vector Embedding Models for Example Selection in Text-to-SQL Generation" Developed 01/06/2024-23/08/2024

Central repository for University of Edinburgh Computer Science MSc Dissertation project for student B244333.

**Code developed for this dissertation project is built atop the DAIL-SQL study open-source code repository, provided alongside the paper "Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation" by Gao et al. (November 2023).**

**Repository: [https://github.com/BeachWang/DAIL-SQL](https://github.com/BeachWang/DAIL-SQL)**

**Paper: [https://arxiv.org/pdf/2308.15363](https://arxiv.org/pdf/2308.15363)**

Access and modifications to the original codebase is granted under the terms of the Apache License 2.0 for private educational use.

**Reccomended Installation: Python 3.11.7**

## Benchmark Installation

- The [Spider](https://yale-lily.github.io/spider) benchmark should be downloaded and upzipped to the location `benchmarks/spider` relative to the root directory. 

    [Alternative Link](https://drive.google.com/file/d/1BM9R1ZIVMRK6UqtKv5UejqfDr5xrBNIp/view?usp=drive_link) (Spider database link appears to be dead as of 16/08/2024).

- Similarly the [BIRD](https://bird-bench.github.io/) directory should also be downloaded and unzipped to the location `benchmarks/bird` relative to the root directory. Both train and dev set folders should be downloaded from the BIRD website and extracted to the same `benchmarks/bird` parent directory. The zip files containing `dev_databases` and `train_databases` must also be extracted before running the pre-processing script with the contents being directly placed into the `databases` directory. By doing this manually, this will dramatically speed up the pre-processing step and is heavily encouraged. the `databases` directory should be filled with many subdirectories that each have the name of a BIRD benchmark database by the end of this setup. 


Upon installing the .zip files, the directory structure should be as follows:
```
benchmarks/
├── spider/
│   ├── databases/
│   ├── test_data/
│   ├── test_database/
│   ├── dev_gold.sql
│   ├── dev.json
│   ├── tables.json
│   ├── train_gold.sql
│   ├── train_others.json
│   ├── train_spider.json
│
└── bird/
    |── databases/
    |   |── ... many database subdirectories
    |   |── ... additional DS_STORE and JSON source files from dev_databases and train_databases.
    ├── dev/
    │   ├── dev_tables.json
    │   ├── dev_tied_append.json
    │   ├── dev.json
    │   └── dev.sql
    └── train/
        ├── train_gold.sql
        ├── train_tables.json
        └── train.json
```

## Environment Setup

- In this project, the Python modules and packages are organized relative to the root directory of the project, which should be `s2593817-codebase/`. To easily configure this in a VSCode IDE environment, insert `"jupyter.notebookFileRoot": "${workspaceFolder}"` into the settings.json file, this can be accessed by entering `Preferences: Open User Settings (JSON)` into the command palette. This should resolve any relative import issues encountered when running Jupyter notebook cells.

- To be able to conduct some parts of the schema-linking script  [stanford-cornlp](https://stanfordnlp.github.io/CoreNLP/) must be downloaded and unzipped to the folder `./third_party`. This requires the installation of the java-jdk, which can be installed via `apt install default-jre` followed by `apt install default-jdk`.


- The list of required packages can be found at `docs/requirements.txt` It is reccomended that before any code is executed, all pre-requisite library installs are first completed which can be quickly achieved by the cmd command ```pip install -r docs/requirements.txt```.*

- Once the necessary packages are installed to the local virtual environment, a pre-processing python script **must** be executed from the command line before any Jupyter notebooks are accessed. This code is used to generate Spider and BIRD dataset objects, and is executed via  `python .\utils\data\data_preprocess.py`.* 

    This is a lengthy script that derives from the DAIL-SQL codebase with small alterations made that are necessary for this study. It only requires being run once, and takes approximately 2 hours.

- _*Note: On Windows machines, you might need to run `$env:PYTHONPATH = (Get-Location).Path`, or on macOS, `export PYTHONPATH=$(pwd)`, to correctly set up relative path referencing from the project root._


## Experiment Notebooks

The directory structure of the codebase is aligned to the main dissertation PDF sumbmission, with all experiments conducted in independent Jupyter notebooks. Experiments leverage the `gpt-3.5-turbo` model through the OpenAI API, this requires the use of an OpenAI account and paid-access tokens. Setting the API access keys should be completed before attempting to run any experiment notebook, this can be completed by setting the `API_KEY` variable in the `OPENAI_API_KEYS` class the top of the `utils/parameters.py` file. After setting this variable to a valid OpenAI API key, experiments become able to communicate with OpenAI LLMs. The `PROJECT_ID` and `ORG_ID` variables may also be set accordingly.

All experiments are fully reproducible by executing 'Run All' at the top of each notebook. For general experiments in `chapter-3` and `chapter-4`, any cell should be executable after running the first three data loading / library import cells at the top of the file. Individual results files are also included in the submission. It is recommended to review the generated prompts in the experiments' results.json files to compare the quality of example retrieval across the various approaches evaluated.

## Directory Contents
- **appendicies:** Contains the experiment notebook, fine-training script, and final results of the question embedding supervised fine-tuning experiment mentioned in Section 4.2 and fully outlined in Appendix D.
- **benchmarks:** Should contain the 'spider' and 'bird' subdirectories once downloaded and extracted from the zip files. These store all of the training files, tests sets and databases included in the respective benchmarks.
- **chapter-3:** Contains all experiment notebooks, results, and analysis files for data presented in Chapter 3 of the dissertation document: "RQ1: Survey of Vector Embedding Model Performance". This has all of the source code used to conduct the survey of eight state-of-the-art vector embedding models for Spider and BIRD benchmarks.
- **chapter-4:** Contains all experiment notebooks, results and analysis files for data presented in Chapter 4 for the dissertation document: "RQ2: Supervised Fine-Tuning of Vector Embedding Models". This has all the source code used to produce the `sft-sql-embedding` fine-tuned embedding model, as well as the notebooks/results for Experiments 1-6 as described in Section 4.4 "Embedding Fine-Tuning-Experiment".
- **data:** Contains copies of the training data for SFT and cached OpenAI embeddings to avoid repeated computations of paid-access sentence embeddings. This also contains the pre-prediction files used in the DAIL-SQL study and incorporated into the experiments of Chapter 4. 
- **models:** This directory is not included in the submission as the fine-tuned embedding models that were produced for the study have now been published to the huggingface platform. The two fine-tuned embeddings are available at: 
    - `sft-question-embedding`: https://huggingface.co/s2593817/sft-question-embedding 
    - `sft-sql-embedding`: https://huggingface.co/s2593817/sft-sql-embedding

    If the model training scripts are run locally to reproduce these fine-tuned models, they will be output to the 'models' subdirectory.
- **third\_party:** Code that has been sourced from external repositories that are not distributed as a part of the DAIL-SQL base repository are placed in third_party. This includes:

    - stanford-core-nlp: Language processing software used in RAT-SQL database schema-linking / pre-processing. (Source: https://stanfordnlp.github.io/CoreNLP/)
    - TSED: Script to compute the tree similarity of edit distance metric that is used as a component of *SQLSim* (Source: https://github.com/Etamin/TSED)
    - Spider Evaluation Script: The code used to evaluate the EX score for Spider benchmark attempts (Source: https://github.com/taoyds/test-suite-sql-eval)
    - BIRD Evaluation Script: The code used to evaluate the EX score for Spider benchmark attempts (Source: https://github.com/bird-bench/mini_dev)
- **utils:** Code predominantly sourced from the DAIL-SQL study which forms the main pipeline for prompt construction and response generation from a GPT model. Significant edits are made to `utils\prompts\ExampleSelectorTemplate.py` to define new example selection algorithms. Any util functions produced for the purposes of this dissertation project are stored in `utils\utils.py`, notably including the `SQLMask` function that is described in Section 2.6.2 of the report.
- **vector_cache:** This is directory is not included in the submission, and is created as a byproduct of running the data_preprocess.py script.

## Acknowledgements

The code produced in this repository is produced for consideration toward the University of Edinburgh Computer Science MSc. I'd like to acknowledge the guidance of my supervisors, and support staff within the UOE Department of Informatics for continued support across the project.