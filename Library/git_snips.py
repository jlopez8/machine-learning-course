
# Steps for Hiding Jupyter Notebook Outputs from Git while preserving locally
1. Add a filter to git config by running the following command in bash inside the repo:`git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'`

2. Create a .gitattributes file inside the directory with the notebooks

3.Add the following to that file:
` *.ipynb filter=strip-notebook-output`  

