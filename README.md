<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/svg/id3-depth-limit-logo-on-dark.svg">
    <img src="docs/brand/svg/id3-depth-limit-logo-on-light.svg" alt="ID3 with Depth Limit" width="260">
  </picture>
</p>

<h1 align="center">Grow a decision tree. Stop at a depth you choose.</h1>

<p align="center">
  Archived university homework in Python, July 2020.<br>
  <b>ID3 from scratch. Logistic regression takes over at the depth limit.</b>
</p>

<p align="center">
  <img alt="Status: archived" src="https://img.shields.io/badge/status-archived-2F4A3A?style=flat-square">
  <img alt="Written in 2020" src="https://img.shields.io/badge/written-2020-1E2A38?style=flat-square">
  <img alt="Python 3" src="https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Four sample datasets" src="https://img.shields.io/badge/sample%20datasets-4-1E2A38?style=flat-square">
  <a href="https://github.com/mertkayacs/ID3-with-Depth-limit"><img alt="GitHub stars" src="https://img.shields.io/github/stars/mertkayacs/ID3-with-Depth-limit?style=flat-square&logo=github&color=C4A772"></a>
</p>

<p align="center">
  <a href="#quick-start"><img alt="Run it" src="https://img.shields.io/badge/Run%20it-2F4A3A?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48ZyBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNMTIgMTloOCIvPjxwYXRoIGQ9Im00IDE3IDYtNi02LTYiLz48L2c%2BPC9zdmc%2B&logoColor=white"></a>
  <a href="hw2.py"><img alt="Source" src="https://img.shields.io/badge/Source-1E2A38?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48ZyBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJtMTYgMTggNi02LTYtNiIvPjxwYXRoIGQ9Im04IDYtNiA2IDYgNiIvPjwvZz48L3N2Zz4%3D&logoColor=white"></a>
  <a href="https://mertkayacs.com/?utm_source=github&utm_medium=readme&utm_campaign=ID3-with-Depth-limit-repo&utm_content=portfolio-badge"><img alt="Portfolio" src="https://img.shields.io/badge/Portfolio-1E2A38?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48ZyBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIyIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik0xMiAyYTE0LjUgMTQuNSAwIDAgMCAwIDIwYTE0LjUgMTQuNSAwIDAgMCAwLTIwTTIgMTJoMjAiLz48L2c%2BPC9zdmc%2B&logoColor=white"></a>
</p>

ID3 with Depth Limit builds a decision tree on categorical data using entropy and information gain. When the tree reaches the depth you set, the node becomes a logistic regression classifier fitted on the rows that arrived there. The script scores the tree with k-fold cross validation and prints the accuracy next to a Naive Bayes classifier written in the same file. It was homework, committed between 6 and 8 July 2020.

---

## Why ID3 with Depth Limit

One script contains the tree, the baseline and the evaluation.

|  |  |
|---|---|
| <img src="docs/assets/readme/icons/git-fork.svg" width="18" height="18" alt=""> **ID3 written by hand** | Entropy, information gain and the tree are plain Python in [hw2.py](hw2.py). scikit-learn supplies only the logistic regression at the depth limit. |
| <img src="docs/assets/readme/icons/chart-no-axes-column.svg" width="18" height="18" alt=""> **A baseline in the same run** | The script prints the tree accuracy as `DTLog` and the Naive Bayes accuracy as `NB`, both from k-fold cross validation. |
| <img src="docs/assets/readme/icons/database.svg" width="18" height="18" alt=""> **Make your own data** | [data-generator2.py](data-generator2.py) writes a synthetic CSV with as many features and samples as you ask for. |

## Features

**Model**

- ID3 decision tree with an optional depth limit.
- Logistic regression at the depth limit.
- Naive Bayes classifier for comparison.

**Data**

- data.csv with 1000 rows and four features, plus three small CSV files for quick tests.
- A generator for synthetic datasets with a binary target.

## Quick start

```sh
pip install pandas scikit-learn
python3 hw2.py data.csv unused 3
```

The first argument is the CSV file (first column an index, last column the target). The second argument is not read by the script, so any word works. The third is the depth limit. Leave out the second and third to grow the tree without a limit.

The script prints two lines, DTLog for the tree and NB for Naive Bayes. To make a dataset, pass the number of samples and the number of values per feature:

```sh
python3 data-generator2.py 20 3 3 4
```

### Requirements

- Python 3.
- pandas and scikit-learn.

## FAQ

<details>
<summary><b>What happens at the depth limit?</b></summary>

The node becomes a logistic regression model fitted on the rows that reached it, instead of splitting further.
</details>

<details>
<summary><b>How is it evaluated?</b></summary>

With k-fold cross validation, next to a Naive Bayes classifier implemented in the same script.
</details>

<details>
<summary><b>Which file do I run?</b></summary>

hw2.py. [ID3depthlimited.py](ID3depthlimited.py) is the earlier version of the same code and needs the depth argument.
</details>

<details>
<summary><b>What format does the data need?</b></summary>

A CSV with an index in the first column, categorical feature columns, and the target in the last column.
</details>

If ID3 with Depth Limit is useful to you, a ⭐ helps other people find it.

[![views](https://hits.sh/github.com/mertkayacs/ID3-with-Depth-limit.svg?view=today-total&style=flat-square&label=views&color=2F4A3A&labelColor=1E2A38)](https://github.com/mertkayacs/ID3-with-Depth-limit)
