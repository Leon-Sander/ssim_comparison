#### SSIM Comparison

We noticed that different implementations of the structural similarity index metric and the multi scale SSIM
have different results. This code calculates the metrics from different implementation to compare the results.

#### Covered Implementations
https://github.com/VainF/pytorch-msssim/
https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
https://www.tensorflow.org/api_docs/python/tf/image/ssim
https://mubeta06.github.io/python/sp/_modules/sp/ssim.html
http://vision.cs.stonybrook.edu/~kema/docwarp/eval.zip

#### Running the Code
Change the img file paths in the ssim_comparison.py file and run it.
The results will be logged in a logging file, which you can specify aswell.