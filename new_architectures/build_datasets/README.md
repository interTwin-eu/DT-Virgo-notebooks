# Torchvision problems and fixes
The current environment presents a huge pain.
Due to incompatible `pytorch`, CUDA, `torchvision` and `pillow` combinations, we had to manually fix a small issue.

`pillow` renamed the variable `PILLOW_VERSION` to `__version__`.
This breaks `torchvision` at file `/opt/conda/envs/pytorch/lib/python3.7/site-packages/torchvision/transforms/functional.py`, located side to the the env root pointed by `sys.executable`.
The easy fix we used was adding a line at the top of such file containing `from PIL import __version__ as PILLOW_VERSION`.

This makes everything work.
The ideal solution is moving to a newer python and CUDA set, which is currently out of my scope.
