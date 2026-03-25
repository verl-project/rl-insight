Welcome to RL-Insight's documentation!
========================

RL-Insight is a performance insight and observability toolkit for reinforcement learning (RL) systems.
It focuses on analyzing offline cluster logs first, and is evolving towards full-stack online monitoring
and visualization.

--------------------------------------------

.. _Contents:

.. toctree::
   :maxdepth: 2
   :caption: Overview

   insight_framework
   cluster_analysis

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   README <../README.md>

.. toctree::
   :maxdepth: 2
   :caption: Data Directory

   data/data_directory

Contribution
-------------

RL-Insight is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/verl-project/rl-insight>`_ for discussions.

Contributions from the community are welcome! Please check out our `project roadmap <https://github.com/verl-project/rl-insight/issues/6>`_ and `good first issues <https://github.com/verl-project/rl-insight/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`_ to see where you can contribute.

Code Linting and Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use pre-commit to help improve code quality. To initialize pre-commit, run:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

To resolve CI errors locally, you can also manually run pre-commit by:

.. code-block:: bash

   pre-commit run

Adding CI tests
^^^^^^^^^^^^^^^^^^^^^^^^

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file.
2. Add related path patterns to the ``paths`` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).