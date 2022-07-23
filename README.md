# Snippet Policy Network

This repository is the official implementation of [Snippet Policy Network V2: Knee-Guided Neuroevolution for Multi-Lead ECG Early Classification](https://ieeexplore.ieee.org/document/9825701), published at IEEE Transactions on Neural Networks and Learning Systems. Feel free to contact me via this email (yuvisu.cs04g@nctu.edu.tw) if you get any problems.


### Guideline:

1. Clone this repository.
2. Run get_data.sh
3. Run the main program to train a model.
4. Use the Online_Inference to evaluate the model.

* Make sure that you have set the correct path when running the code.

We provide a well-trained model (the model of the first fold in 10-fold cross-validation) to validate the performance in our paper.

### If you find this code helpful, feel free to cite our paper:
```
@ARTICLE{Huang2022SPNV2,
  author={Huang, Yu and Yen, Gary G. and Tseng, Vincent S.},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Snippet Policy Network V2: Knee-Guided Neuroevolution for Multi-Lead ECG Early Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2022.3187741}}
```