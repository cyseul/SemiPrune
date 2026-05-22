# SemiPrune

This repository provides an implementation of SemiPrune, a label-efficient dataset pruning framework that enables existing supervised pruning methods to be applied without full annotation.

SemiPrune first uses a small randomly labeled subset to train a semi-supervised learning model and generate pseudo-labels for unlabeled data. It then applies supervised pruning methods to the resulting pseudo-labeled training pool.
