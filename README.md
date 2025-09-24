# PointAnno-Penumonia
Here is the project response: "Point-annotation supervision for robust 3D pulmonary infection segmentation by CT-based cascading deep learning". If you encounter any questions, please feel free to contact us. You can create an issue or just send an email to mengzi98@163.com or yuetan.chu@kaust.edu.sa. Also welcome for any idea exchange and discussion.

## Installation
```
conda create -n segmentation python==3.8
conda activate segmentation
pip install -r requirements.txt
```
Here we include all the packages used in our whole platform. However, some packages are not used in this project. You can install some of these packages according to your situation.

## Sample data
A part of the accessible data (PointP) (n=20) and the full-annotation of infections can be downloaded [here](https://drive.google.com/drive/folders/1ZuEn3Uq9AFiy7mM4Q-mgljAFfeTuNTR6?usp=sharing). Feel free to use these data for training and testing your segmentation model. If you want to access more data, please do not hesitate to contact mengzi98@163.com.

### Workflow
![image](https://github.com/Arturia-Pendragon-Iris/Point-annotation-segmentation/blob/main/img/Figure_2.png)

### Performance evaluation
![image](https://github.com/Arturia-Pendragon-Iris/Point-annotation-segmentation/blob/main/img/Figure_6.png)

## Acknowledge
* [CycleGAN](https://github.com/charlesyou999648/GAN-CIRCLE) The development of CT inter-slice super-resolution is based on this repo.
* [Blood vessel segmentation](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation)
* [Lung and airway segmentation](https://github.com/LongxiZhou/DLPE-method)
* [CT denoising](https://github.com/SSinyu/RED-CNN)

## Cite
You can cite our paper as follows:
```
@article{chu2025point,
  title={Point-annotation supervision for robust 3D pulmonary infection segmentation by CT-based cascading deep learning},
  author={Chu, Yuetan and Wang, Jianpeng and Xiong, Yaxin and Gao, Yuan and Liu, Xin and Luo, Gongning and Gao, Xin and Zhao, Mingyan and Huang, Chao and Qiu, Zhaowen and others},
  journal={Computers in Biology and Medicine},
  volume={187},
  pages={109760},
  year={2025},
  publisher={Elsevier}
}
```



