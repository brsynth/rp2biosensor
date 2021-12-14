# rp2biosensor -- Build Sensing-Enabling Metabolic Pathways from RetroPath2.0 output

rp2biosensor extracts metabolic paths linking undetectable compounds
into a detectable one using biochemical reactions. This concept is known
as "Sensing-Enabling Metabolic Pathways". See [doi: 10.1021/acssynbio.5b00225](https://doi.org/10.1021/acssynbio.5b00225)
for more.

In short, rp2bionsensor converts the output of [RetroPath2.0](https://www.myexperiment.org/workflows/4987.html) into a HTML page showing the possible metabolic paths linking the compound to be detected to the detectable ones.

## Install

```bash
conda install -c conda-forge rp2biosensor
```

## Run

```bash
python -m rp2biosensor /path/to/rp2/results.csv --opath /path/to/output/file.html --otype file
```

By default, all the needed dependancies are embedded into the HTML file. This includes CSS, JavaScript, and data files. Alternatively, the output could be outputted into a directory, using the `dir` output type:

```bash
python -m rp2biosensor /path/to/rp2/results.csv --opath /path/to/output --otype dir
```

The embedded help:
```bash
python -m rp2biosensor -h

usage: rp2biosensor [-h] [--opath OPATH] [--otype {dir,file}] rp2_results

Generate HTML outputs to explore Sensing Enabling Metabolic Pathway from RetroPath2 results.

positional arguments:
  rp2_results         RetroPath2.0 results

optional arguments:
  -h, --help          show this help message and exit
  --opath OPATH       Output path. Default: /Users/tduigou/code/rp2net/biosensor.html.
  --otype {dir,file}  Output type. This could be either (i) "dir" which means ouput files will outputted into this directory, or (ii) "file" which means that all files will be embedded into a single HTML page. Default:
                      file
```

## Example

```bash
python -m rp2biosensor tests/data/input/rp2-results_dmax-16.csv --opath ./biosensor.html
```

## For developpers

### Install
```bash
conda env create -f environment.yaml -n rp2biosensor-dev
conda develop -n rp2biosensor-dev .
```

### Test
```bash
python -m pytest -vv
```
