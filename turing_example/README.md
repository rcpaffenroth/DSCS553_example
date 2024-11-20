# Commands to try

jupyter nbconvert --to notebook --execute quickstart_tutorial.ipynb --output=quickstart_tutorial_output.ipynb --ExecutePreprocessor.timeout=-1

jupytext --to py quickstart_tutorial.ipynb

wandb sweep sweep_runner.yml

