Welcome to RL-Insight's documentation!
======================================

RL-Insight is a performance insight and observability toolkit for reinforcement learning (RL) systems.
It focuses on online observability for distributed RL training.

Start with :doc:`Quick Start <tutorial/quick_start>` to verify metrics and traces,
then use :doc:`Configuration <programming_guide/interfaces/configuration>` to configure your deployment.

--------------------------------------------

.. _Contents:

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   Quick Start <tutorial/quick_start>
   Server Installation <tutorial/server_installation>

.. toctree::
   :maxdepth: 2
   :caption: Programming Guide

   Python API <programming_guide/interfaces/python_api>
   Configuration <programming_guide/interfaces/configuration>
   CLI Command <programming_guide/interfaces/cli_command>
   Environment Variable <programming_guide/interfaces/environment_variable>
   Agent Loop Protocol <programming_guide/agent_loop_protocol>
   Metric Name Catalog <programming_guide/metric_name_catalog>

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features

   Hardware Monitoring <advanced_features/hardware/index>
   Local Persisted Data Status <advanced_features/local_persisted_data_status>
   Distributions in Push Sinks <advanced_features/push_distributions>
   Custom Grafana Dashboards <advanced_features/custom_grafana_dashboards>
   Data Directory Migration <advanced_features/data_directory_migration>

.. toctree::
   :maxdepth: 1
   :caption: Framework Support

   Verl <framework_support/verl>
   Verl-omni <framework_support/verl_omni>
   Verl-SpeCo <framework_support/verl_speco>
   Slime_ascend <framework_support/slime_ascend>

.. toctree::
   :maxdepth: 1
   :caption: Tuning Cases

   Async RL Training Tuning Guide <tuning_cases/async_rl_training_tuning_guide>
   Multimodal Model Tuning Using RL-Insight <tuning_cases/multimodal_model_tuning_using_rl_insight>

.. toctree::
   :maxdepth: 1
   :caption: Other

   Stress Test Report <other/stress_test_report>

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
