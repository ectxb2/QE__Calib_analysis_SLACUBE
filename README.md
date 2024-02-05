# SLACube_PECalib

## Set up the environment

Modify `setup.sh` script according to your environment.
For `SLACUBE_LAYOUT`, you can find the SLACube layout file at 
`/sdf/data/neutrino/yuntse/slacube/etc/layouts/slacube_layout-2.4.0.yaml`
on S3DF.

```shell
source setup.sh
```

Created on Thu Oct  5 21:23:52 2023

@author: yuntse
Also includes work by Dan Douglas and Liz Triller

First: source setup.sh
This may need to be edited to reflect your path to the files

then:
python3 bin/analyzePESignalMax_val.py -c "one_cfg" -s exttrig_2023_10_03_02_28_06_PDT.h5 -b exttrig_2023_10_03_03_05_54_PDT.h5 -t 'new try3'


## Laser/LED calibration data

Both `bin/analyzePESignal.py` and `bin/analyzePESignalList.py` 
calculate the mean value and the standard deviation
of each pixel channel, subtract the background (the
mean value of the background data),
save the result in the h5 format, and make plots.
`bin/analyzePESignal.py` handles a single signal and a background files, 
while `bin/analyzePESignalList.py` deals with filelists
containing multiple signal and background files.
Run
```shell
python bin/analyzePESignalList.py -h
```
to get the instruction.
