The Scruples Dilemmas
=====================
This directory contains the Scruples Dilemmas.

Scruples provides code and data for understanding norms and subjectivity in
natural language. For more information, see the code repository
(https://github.com/allenai/scruples), or the paper: "Scruples: A Corpus of
Community Ethical Judgments on 32,000 Real-Life Anecdotes"
(https://arxiv.org/abs/2008.09094).

To cite this data or other aspects of the paper, use:

    @article{Lourie2020Scruples,
        author = {Nicholas Lourie and Ronan Le Bras and Yejin Choi},
        title = {Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life Anecdotes},
        journal = {arXiv e-prints},
        year = {2020},
        archivePrefix = {arXiv},
        eprint = {2008.09094},
    }


Data Structure
--------------
Each dilemma from the Scruples Dilemmas has the following attributes:

  - **id**
    A unique identifier for the dilemma.
  - **actions**
    An array containing the two actions presented in the dilemma. Each action
    has the following attributes:
    - **id**
      A unique identifier for the action, matching the ID for the anecdote from
      which the action originates in the Scruples Anecdotes.
    - **description**
      The textual description of the action.
  - **gold_annotations**
    The counts for the gold label annotations from Mechanical Turk. The first
    element of the array is the number of annotators that chose the first
    action as *less ethical*. The second element is the number of annotators
    that chose the second action as *less ethical*.
  - **gold_label**
    The majority label for the gold annotations.
  - **human_perf_annotations**
    Additional annotations from Mechanical Turk, crowd sourced for the purpose
    of measuring human performance. They're only non-zero on the dev and test
    splits. Human performance annotations are provided on dev to enable error
    analyses that compare model predictions with human predictions.
  - **human_perf_label**
    The majority label for the human performance annotations.
  - **controversial**
    `true` if the gold label annotations were not unanimous, otherwise `false`.


Files
-----
This directory should contain the following files:

  - **README**
    This README file.
  - **train.scruples-dilemmas.jsonl**
    The training split.
  - **train-extra.scruples-dilemmas.jsonl**
    Extra training data annotated by workers before and during qualification,
    so it has a slightly different data distribution.
  - **dev.scruples-dilemmas.jsonl**
    The development split.
  - **test.scruples-dilemmas.jsonl**
    The testing split.


Contact
-------
For more information, see the code repository
(https://github.com/allenai/scruples), or the paper: "Scruples: A Corpus of
Community Ethical Judgments on 32,000 Real-Life Anecdotes"
(https://arxiv.org/abs/2008.09094). Questions and comments may be addressed to
Nicholas Lourie.
