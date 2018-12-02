# Perlin Adversarial Examples

This repository contains sample code and an interactive Jupyter notebook for the paper ["Procedural Noise Adversarial Examples for Black-Box Attacks on Deep Neural Networks"](https://arxiv.org/abs/1810.00470).

Procedural noise functions are parametrized and used to generate textures in computer graphics. In this work we use Perlin noise, a type of procedural noise, to create adversarial perturbations against popular deep neural network architectures trained on the ImageNet image classification task.

The results show that adversarial examples can be generated using Perlin noise **without any knowledge of the target classifier.** This demonstrates the instability of current neural networks to procedural noise patterns.

You can play with the noise function parameters to make your own adversarial examples with our interactive widget in the Jupyter notebook.

![slider](slider.png)

Please see our [paper](https://arxiv.org/abs/1810.00470) for more details: "Procedural Noise Adversarial Examples for Black-Box Attacks on Deep Neural Networks." Kenneth T. Co, Luis Muñoz-González, Emil C. Lupu. arXiv 2018.

## Python Dependencies

* ipywidgets
* jupyter
* keras
* matplotlib >= 2.0.2
* noise >= 1.2.0
* numpy
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