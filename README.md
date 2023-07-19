# ODAM:Gradient-based instance-specific Visual Explanation for Object Detection

ODAM is a straightforward and easy-to-implement method to generate visual explanation heat maps for predictions of object detection. The framework and results are shown here:

<img width=90% src="https://github.com/Cyang-Zhao/ODAM/blob/main/images/framework.jpg"/>
<img width=90% src="https://github.com/Cyang-Zhao/ODAM/blob/main/images/examples.jpg"/>

# Example of Generating visual explanation maps by ODAM:

ODAM is easy to be applied on different detector architectures. Here is an example:
- Detector: FCOS
- Data: MS COCO val2017 
- Demo for one image: [Demo_ODAM](https://github.com/Cyang-Zhao/ODAM/blob/main/tools/demo_ODAM.ipynb) ; Demo based on DETR minimal implementation: [Demo_ODAM_detr](https://colab.research.google.com/drive/1j-JZuZ3FXXQucr_LZWSxCUmHsqpWzaVR#scrollTo=kqe_0nc5dyAq)
  

Steps to save heat maps and evaluation:
1. The path of the dataset is set in `config_coco.py`.
2. Download the fcos detector [model](https://www.dropbox.com/s/v70pq3x5w74yenn/dump-12.pth?dl=0) and put into the folder `./model/fcos_regular/coco_model/`; faster rcnn detector [model](https://www.dropbox.com/scl/fi/jbdsi9u6iyjhaqfmd0hgk/dump-12.pth?rlkey=z7892nlzx6qmu6y1ymm1k7a5y&dl=0) and put into the folder `./model/rcnn_regular/coco_model/`.
3. ```cd tools```

4. Saving heat maps for high-quality predictions
- Saving ODAM explanation maps:
```
python savefig_odam.py -md fcos_regular -r 12
```
- Saving D-RISE explanation maps:
```
python savefig_drise.py -md fcos_regular -r 12
```
5. Evaluation of ODAM and D-RISE:
- Point Game
```
python eval_pointgame.py -md fcos_regular -t odam
```
- Visual Explanation Accuracy (Mask IoU)
```
python eval_mask_IoU.py -md fcos_regular -t odam
```
- ODI
```
python eval_odi.py -md fcos_regular -t odam
```
- Deletion
```
python eval_delet.py -md fcos_regular -r 12 -t odam
```
- Insertion
```
python eval_insert.py -md fcos_regular -r 12 -t odam
```


# Odam-Train and Odam-NMS:

Train and test on [CrowdHuman](https://www.crowdhuman.org/) dataset, the data path is set in `config_crowdhuman.py`.
- For train, download the [initial weights](https://www.dropbox.com/s/1yb1s3hacg68cam/resnet50_fbaug.pth?dl=0), and the path is set in `config_crowdhuman.py`, then run:
```
cd tools
python train_crowdhuman.py -md fcos_odamTrain
```
- For test, download the fcos [model](https://www.dropbox.com/s/5oqciysj6ip5tvf/dump-30.pth?dl=0) and put into the folder `./model/fcos_odamTrain/outputs/` and faster rcnn [model](https://www.dropbox.com/scl/fi/gcoasnmq0f4ll7g6ydpfj/dump-30.pth?rlkey=z72ocld13xcsjwzamxv5c7dw7&dl=0) to `./model/rcnn_odamTrain/outputs/`. The NMS method choosing option is set in `config_crowdhuman.py`, then run:
```
cd tools
python test_crowdhuman.py -md fcos_odamTrain -r 30
```

# Citation

If you use the code in your research, please cite:
```
@inproceedings{chenyangodam,
  title={ODAM: Gradient-based Instance-Specific Visual Explanations for Object Detection},
  author={Chenyang, ZHAO and Chan, Antoni B},
  booktitle={The Eleventh International Conference on Learning Representations},
  month = {May},
  year = {2023}
}
```

# Contact

If you have any questions, please do not hesitate to contact Chenyang ZHAO (zhaocy2333@gmail.com).
