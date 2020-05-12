# How to contribute

## Propose a new model
The scope of this library is to provide a variety of neural network models that can perform dimensionality reduction.

If you want to propose a new model, add a new issue with the [PROPOSITION] prefix in the issue name. State the name and a short description of the model and provide a link to the paper or other research document that describes it.
When we accept the new model the label will change to `implement` and the new model will be implemented as soon as possible.

## Reporting a bug
When you encounter a bug, try to provide a clear description of the problem so it is easy to reproduce and fix :)

## Making a change
If you feel ready to start adding or modifying code, make sure you clearly state what you are working on. Then fork and make your changes. Afterwards you can create a pull request. If all checks have passed and a code review has been performed your changes will be merged into master.

## Coding conventions
- We are using the PEP 8 codestyle, this is enforced via pycodestyle so any pull request will be automatically checked
- Documentation is written in the numpydoc style
- Each model requires a new file, if the model uses new layers, regularizers or other parts. Then they also require new files.
