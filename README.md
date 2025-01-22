# Histogram Transfer

WIP repo for testing different 3D Histogramm transfer methods.
- `statistical_analysis.py` is a script used to compute covarainces between channels for different color spaces. The objective was to find independent channels to apply single channel histrogram transfer (and check that Ruderman et al. method generates non correlated channels). Its `main` function explores the `data` folder, reads images, projects them in the desired colorspace and computes the covariances. Outputs a pandas DataFrame with the desired info. 
- `ordering_methods.py` contains different methods for histogram transfering. Most of them are just for experiment purposes, so they are neither optimized nore relevant.
- `methods_comparison.ipynb` is a notebook used to load images and test transfer methods
- `data` is a folder containing differnet RGB images for testing
