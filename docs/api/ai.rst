AI Module
=========

A language model for ACTUNEO: one that runs on the user's own machine, chosen
to suit its hardware, or a provider reached with the user's own key.

The local model needs the optional packages::

   pip install actuneo[ai]

.. automodule:: actuneo.ai
   :no-members:

Setting up
----------

.. autofunction:: actuneo.ai.setup

.. autofunction:: actuneo.ai.status

.. autofunction:: actuneo.ai.remove_model

.. autofunction:: actuneo.ai.configure

.. autofunction:: actuneo.ai.get_backend

How a model is chosen
---------------------

A model fits when its memory need, with headroom left for the operating
system and Python, is within what is free:

* on an NVIDIA card, within free video memory less 20%, with every layer on
  the card;
* on Apple Silicon, within free memory less 30% (the GPU shares it);
* otherwise within free memory less 30%, on the processor. An NVIDIA card
  too small for the whole model takes as many layers as it can hold.

The download plus 1 GB must also fit on the disk. The recommended model is
the largest that fits, except that a model which cannot sit wholly on a GPU
is recommended only up to about 5 GB, because larger ones answer slowly on a
processor. Those are still offered as the "largest" alternative, which allows
only 10% headroom. If memory cannot be measured the smallest model is taken,
and if nothing fits no model is downloaded.

.. autofunction:: actuneo.ai.select_model

.. autofunction:: actuneo.ai.detect_hardware

.. autoclass:: actuneo.ai.HardwareProfile
   :members:

The catalog
-----------

.. automodule:: actuneo.ai.catalog
   :no-members:

.. autofunction:: actuneo.ai.load_catalog

.. autoclass:: actuneo.ai.ModelSpec
   :members:

.. autofunction:: actuneo.ai.download_model

Backends
--------

.. automodule:: actuneo.ai.backends
   :no-members:

.. autoclass:: actuneo.ai.Backend
   :members:

.. autoclass:: actuneo.ai.ToolCall

.. autoclass:: actuneo.ai.LocalBackend

.. autoclass:: actuneo.ai.OpenAIBackend

.. autoclass:: actuneo.ai.AnthropicBackend

.. autoclass:: actuneo.ai.GeminiBackend

.. autoclass:: actuneo.ai.OllamaBackend

.. autofunction:: actuneo.ai.tool_choice_schema

Where files are kept
--------------------

.. automodule:: actuneo.ai.config
   :no-members:
