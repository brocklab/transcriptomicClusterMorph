# Deep Learning Identification of Heterogeneous Subpopulations 
This is a repository to archive code from the paper [Deep learning identifies heterogeneous subpopulations in breast cancer cell lines](https://www.biorxiv.org/content/10.1101/2024.07.02.601576v1).

## Please note that these models will need to be updated for your dataset

Data and models are available upon request. 

The requirements for this repository are a bit complex, but in the following order:
0. Make a conda environment for analysis
1. Install PyTorch >= 1.8
2. Build detectron2 from source - https://detectron2.readthedocs.io/en/latest/tutorials/install.html
3. Install OpenCV. Verify proper installation by importing in Python. If you have trouble with this try installing with conda and making sure system dependencies are installed.
4. Install the local package from this repository by calling `pip install -e .`

Sorry that took so long, never lock yourself into using detectron2.

Most notebooks in the primary `/notebook` folder center around prediction (training) or testing (validation). 

Figure generation is located in `/notebooks/publicationFigures`. Recall that we analyze:
- Treated MDA-MB-231 and untreated MDA-MB-231 populations
- MDA-MB-231 transcriptomic subpopulations
- MDA-MB-436 transcriptomic subpopulations

So testing results and figures will generally have subpop/treated or 231/436 in their title.

