# ODAM:Gradient-based instance-specific Visual Explanation for Object Detection

The method framework and results are shown here:

<img width=60% src="https://github.com/Cyang-Zhao/ODAM/blob/main/images/framework.jpg"/>
<img width=90% src="https://github.com/Cyang-Zhao/ODAM/blob/main/images/examples.jpg"/>

# Example of Generating visual explanation maps by ODAM:

- Detector: FCOS
- Model: FCOS-resnet50
- Data: MS COCO val2017 
- Demo for one image: [Demo_ODAM](https://github.com/Cyang-Zhao/ODAM/blob/main/tools/demo_ODAM.ipynb)
- Steps to save heat maps and evaluation:
1. The path of the dataset is set in `config_coco.py`.
2. Download the fcos detecot model and put into the folder `./model/fcos_regular/coco_model/`
3. ```cd tools```
4. 
-Saving ODAM explanation maps:
```python savefig_odam.py -md fcos_regular -r 12```
-Saving D-RISE explanation maps:
```python savefig_drise.py -md fcos_regular -r 12```
5. Evaluation ODAM and D-RISE:
-point game
```python eval_pointgame.py -md fcos_regular -t odam```
-visual explanation accuracy (mask IoU)
```python eval_mask_IoU.py -md fcos_regular -t odam```
-ODI
```python eval_odi.py -md fcos_regular -t odam```
-deletion
```python eval_delet.py -md fcos_regular -r 12 -t odam```
-insertion
```python eval_insert.py -md fcos_regular -r 12 -t odam```


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
