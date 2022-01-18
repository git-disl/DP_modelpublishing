# DP\_modelpublishing

The example code for the paper ["Differentially Private Model Publishing for Deep Learning"](https://arxiv.org/abs/1904.02200)

Some notes:
* The code was based on the original DP-SGD implementation in CCS'16  which was not maitained anymore. It requires Tensorflow 0.10.0. A fork canbe found [here](https://github.com/eric-erki/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/tree/8fcc9a6c9a864834d5941c10b145c0a58ee3d4af/tensorflow_dl_models/research/differential_privacy/dp_sgd)
* In the training code DumpzCDPAccountant is used and the real accounting is done during the training step. zCDPAccountant implementation is in privacy\_accountant/accountant.py but not used yet. 

Directory:
1. The examples with decay functions are in dp\_mnist and dp\_cancerdataset
2. The examples with validation based decay are in dp\_mnist\_validation and dp\_cancerdataset\_validation
3. The privacy\_accountant includes the code for comparing different privacy accountants, and conversion to epsilon, delta DP.

 
