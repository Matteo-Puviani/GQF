# GKPQECFG - Gottesmann-Kitaev-Preskill code quantum error correction with Feedback-GRAPE.
[![arXiv](https://img.shields.io/badge/arXiv-2312.07391-b31b1b.svg)](https://arxiv.org/abs/2312.07391)

Code repository for quantum error correction with non-Markovian Feedback on the Gottesmann-Kitaev-Preskill code with model-based Feedbakc-GRAPE approach.

- [Description](#description)
- [Installation](#installation)
- [License](#license)
- [Citation](#citation)

## Description

The library can be used to train a recurrent or a feed-forward neural network to suggest the parameters of the gate sequence of the small-BIG-small 

<img src="images/scheme_GKP.png" alt="overview" width="800"/>


## Installation

1. Clone the repository

``` bash
git clone https://github.com/Matteo-Puviani/GKPQECFG.git
cd GKPQECFG
```

2. Install requirements
``` bash
pip install -r requirements.txt
```


## License

The code in this repository is released under the MIT License.


## Citation
``` bib
@article{puviani_gkp_2023,
  title={Boosting the Gottesman-Kitaev-Preskill quantum error correction with non-Markovian feedback},
  author={Puviani, Matteo and Borah, Sangkha and Zen, Remmy and Olle, Jan and Marquardt, Florian},
  url = {http://arxiv.org/abs/2312.07391},
  journal={arXiv preprint arXiv:2312.07391},
  publisher = {arXiv},
  month = dec,
  year = {2023},
  note = {arXiv:2312.07391 [quant-ph]},
}
```
