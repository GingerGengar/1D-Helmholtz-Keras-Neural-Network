# Tasks are added in chronological order.

## Given 

-The original script that was given

## NN Archives 

-References on how to change various things

-Not populated, at all

## EvenOdds 

-NN used to represent function and Even and Odd Built into architecture

-Succesful in its implementation. Alternate implementations exists using NN(x) = k 1[f(x) + f(-x)]

## Custom Training 

-Tried to implement loss function

-Failed, problem not well-posed

-Further study need to be conducted on custom-training loops

## Testing Helmholtz 

-First Time testing the different NN parameters

-Manual Testing, works fine

## Constant Width Depth

Refers to a collection of computational experiments for Neural Networks of identical number of nodes per hidden layer (width). The number of hidden layers present is referred as (depth).

### Automation

-First Automation on Constant Width and Depth

-data is a directory made by the helm1d script

-Latex contains the necessary documentations

### Convergence

-Added Convergence Testing to automated testing

-Current Run is a directory used to generate the data

-Inspection Data are just snapshots of different trials

-Post Proc Tools contain the various scripts for post-processing

-Scheduler Trial is an attempt at changing the training scheduler

### Modified Helmholtz

-Confirmed that changes work as intended

### Scheduler

-Combines the changed scheduler with the automation scripts

-Made directories each for polynomial learning rate, inverse learning rate, exponential learning rate

-Added a new post-processing script, 2-Plot which enables to plot 2 simulatenous histories in one graph

-Added a new directory named comparison to compare the constant learning rate with the changed scheduled learning rates
