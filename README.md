# Procedural Adversarial Examples

This repository contains sample code and an interactive Jupyter notebook for the paper ["Procedural Noise Adversarial Examples for Black-Box Attacks on Deep Neural Networks"](https://arxiv.org/abs/1810.00470).

Procedural noise functions are used to generate textures in computer graphics. In this work we present various types of procedural noise to generate adversarial perturbations against popular deep neural network architectures trained on the ImageNet image classification task.

The results show that adversarial examples can be generated using procedural noise **without any knowledge of the target classifier.** This demonstrates the instability of current neural networks to procedural noise patterns.

We encourage you to explore our Python notebooks:

1. **slider_perlin.ipynb** Visualize and interactively play with the parameters to make your own adversarial examples. See the other slider notebooks for more noise functions.
2. **bayes_opt.ipynb** See how Bayesian optimization can automatically find parameters to fool a classifier on an image.

![slider](slider.png)

Please see our [paper](https://arxiv.org/abs/1810.00470) for more details: "Procedural Noise Adversarial Examples for Black-Box Attacks on Deep Neural Networks." Kenneth T. Co, Luis Muñoz-González, Emil C. Lupu. arXiv 2018.

## Python Dependencies

* [GPy](https://pypi.org/project/GPyOpt/)
* [GPyOpt](https://pypi.org/project/GPy/)
* ipywidgets
* Keras
* matplotlib >= 2.0.2
* [noise](https://pypi.org/project/noise/)
* numpy
* [opensimplex](https://pypi.org/project/opensimplex/)
* tensorflow

## Acknowledgments

Learn more about the [Resilient Information Systems Security (RISS)](http://rissgroup.org/) group at Imperial College London. The main author is a PhD student supported by [DataSpartan](http://dataspartan.co.uk/). DataSpartan is not affiliated with the university.

Please cite this paper if you use the code in this repository as part of a published research project.

```
@article{co2018procedural,
  title={Procedural Noise Adversarial Examples for Black-Box Attacks on Deep Neural Networks},
  author={Co, Kenneth T and Mu{\~n}oz-Gonz{\'a}lez, Luis and Lupu, Emil C},
  journal={arXiv preprint arXiv:1810.00470},
  year={2018}
}
```
This project is licensed under the MIT License, see the [LICENSE.md](LICENSE.md) file for details.