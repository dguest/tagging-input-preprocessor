Neural network evaluation example
=================================

This code compares the output from a network evaluated with lwtnn to
the output from keras.

To run the code
---------------

 - make sure you have [git lfs][1] installed
 - run `git clone git@github.com:dguest/tagging-input-preprocessor.git`
 - run `./run.sh`

If anything goes wrong contact me!

Preprocessing the data
-----------------------

 - First run './input_data/flatten_hdf5.py' Make sure you modify the input and output file to the one you want
 - Second run './input_data/split_into_categories.py' This will create the hl_tracks set and the other sets we use to group variables
 - Third, for using the high level features you just have to run './run.sh'. You can change the feature variable to use the group of variables you want
 - The main method in 'evaluate_nn.py' already has the code to subselect the correct features, scale and center the data and make the predictions. You can modify it to make sure it returns the desired values (probabilities or labels)

You can also do all of this at once if you run get_predictions.sh This will save the predicted probabilities and labels into numpy files you can later use to calculate the ROC AUC. Just make sure to pass the correct input file in get_predictions.sh.

Let me know if anything goes wrong!

Requirements
-----------------------
The code runs on python 2.7.13 and 3.6
The package versions for python 2 and python 3 are available in py2_7_freeze.txt and py3_6_freeze.txt respectively


What are all these files
------------------------

The main logic is inside `./test_network.py`, look there for more
explanation.

Briefly, this reads in a trained Keras network along with a file where
the network outputs have already been added. It then compares the two
and prints out any mismatches.

[1]: https://git-lfs.github.com/
