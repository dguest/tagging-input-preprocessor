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

What are all these files
------------------------

The main logic is inside `./test_network.py`, look there for more
explanation.

Briefly, this reads in a trained Keras network along with a file where
the network outputs have already been added. It then compares the two
and prints out any mismatches.

[1]: https://git-lfs.github.com/
