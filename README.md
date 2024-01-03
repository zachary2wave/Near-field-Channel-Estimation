# Near-field Channel Estimation
 

This code corresponds to the code featured in the paper "Near-Field Channel Estimation for Extremely Large-Scale Array Communications: A model-based deep learning approach". 
It is welcome to be applied and referenced. 
If applying, please cite the paper. 

@ARTICLE{zhang2023near,
  author={Zhang, Xiangyu and Wang, Zening and Zhang, Haiyang and Yang, Luxi},
  journal={IEEE Communications Letters}, 
  title={Near-Field Channel Estimation for Extremely Large-Scale Array Communications: A Model-Based Deep Learning Approach}, 
  year={2023},
  volume={27},
  number={4},
  pages={1155-1159},
  doi={10.1109/LCOMM.2023.3245084}}



Here is a brief illustration of the code:

+ The basic parameter is defined in  basic_parameter.py
+ The channel data set is generated in data_generate.py
+ The Neural network model is shown in ISTA_Net(SMO-LISTA) and LISTA(LISTA)
+ The training is proceeded in LISTA_off(LISTA) and M_X_offgridK_128.py(SMO-LISTA)




For any more detail, please contact xy_zhang@seu.edu.cn



基本参数在 basic_parameter.py 中定义
通道数据集在data_generate.py中生成
神经网络模型在ISTA_Net（SMO-LISTA）和LISTA（LISTA）中
训练在LISTA_off(LISTA)和M_X_offgridK_128.py(SMO-LISTA)中进行

