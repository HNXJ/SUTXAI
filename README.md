# TF2 Python classifier for the PhysioNet/CinC Challenge 2020

## Team:SUTXAI

## Use

You can run this classifier by installing the packages in the `requirements.txt` file and running

    python driver.py input_directory output_directory

where `input_directory` is a directory for input data files and `output_directory` is a directory for output classification files. The PhysioNet/CinC 2020 webpage provides a training database with data files and a description of the contents and structure of these files.

The code uses a Python Online and Offline ECG QRS Detector based on the Pan-Tomkins algorithm. It was created and used for experimental purposes in psychophysiology and psychology. You can find more information in module documentation: https://github.com/c-labpl/qrs_detector
