# NLPFinalProject

`data/raw/` contains `data_stage_one` data files from the [convote dataset](http://www.cs.cornell.edu/home/llee/data/convote.html). They are seperated into `train/`, `test/`, and `dev/`. They are separated by speaker, and the political party of the speaker is 'P' in the name of the file: `([0-9]\_)+P\w\w.txt`, where 'R' represents the Republican party, and 'D' represents the Democratic party.

The data can also be downloaded [here](https://www.dropbox.com/s/k5qpdwc2i94l8f0/data.zip?dl=1).

## Running Project

Tag the data with:
```
python tag_raw_dataset.py
```

To run the perceptron on the untagged data, run:
```
python perceptron.py <num_iterations>
```
