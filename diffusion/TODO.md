# List of things to do

* Kyle: look at pytest
* finish an implementation of diffusion operations
* design validation tests for diffusion scripts
* design experiments to check usefulness of various normalization procedures

* make sure existing code works with sparse data structures

* Write a version in cypher that can use cypher functionality to extract paths -- Kyle can help show how to perform matrix normalizations in a manner that is easier to do on cypher.

## Normalization ideas so far

* do we want "juice-preserving" feature (i.e. column stochastic normalization) or not? WHY?
    * want to be able to compare diffusion scores for different metapaths
* what kind of row/column damping parameters do we want?

## For Kyle

* finish module that allows diffusion computation (both forward and backward) with options for all possible normalizations that we're interested in.
* write simple tests