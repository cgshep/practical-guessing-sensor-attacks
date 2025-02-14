[![CI](https://github.com/cgshep/entropy-collapse-mobile-sensors/actions/workflows/python-package.yml/badge.svg)](https://github.com/cgshep/entropy-collapse-mobile-sensors/actions/workflows/python-package.yml)

# Entropy Analyser

This repository is designed to accompany the paper ["Entropy Collapse in Mobile Sensors: The Hidden
Risks of Sensor-Based Security"](https://arxiv.org/pdf/2502.09535).

The work presents entropy values from a range of mobile sensors, such as accelerometers, gyroscopes, magnetometers, and envioronmental sensors.

The results show that sensor data has insufficient unpredictability for any serious security application, resulting in 3.4-4.5 bits for single sensor modalities, and at most ~24 bits when using multiple sensors simultaneously.

Computing these results, particularly joint entropy, is not straightforward. Bayesian networks (Chow-Liu trees) [1,2] are used to decompose direct computation of joint entropies, which suffers from the curse of dimensionality, to something computationally tractable.

Have a look at `analysis.py` and `utils.py` for further details.

## Notes

- Only the UCI-HAR and relay datasets are provided in this repo. The other datasets, PerilZIS and SHL, are VERY large (multiple GBs).
- It should be straightforward to add your own dataset: the estimator requires inputs of the form of a Pandas dataframe with sensor modalities as columns, and rows as sensor values taken simultaneously for those sensors. This usually requires a fair bit of data wrangling; have a look at how it's done for the UCI-HAR dataset.

## License

The entropy analyser is under the MIT License. This does not override the licenses of the datasets.

## References

1. D. Buller and A. Kaufer, ["Estimating min-entropy using probabilistic graphical
models,"](https://csrc.nist.gov/csrc/media/events/random-bit-generation-workshop-2016/documents/abstracts/daryll-buller-full-paper.pdf) Random Bit Generation Workshop, NIST, 2016.
2. C. Chow and C. Liu, ["Approximating discrete probability distributions with dependence trees,"](https://ieeexplore.ieee.org/document/1054142) IEEE Trans. on Information Theory, vol. 14, no. 3, 1968.
