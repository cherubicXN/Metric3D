## Tailored Metric3D for lightweight usage

## 🔨 Installation
### One-line Installation
For the ViT models, use the following environment：
```bash
pip install -r requirements_v2.txt
```

For ConvNeXt-L, it is 
```bash
pip install -r requirements_v1.txt
```

### dataset annotation components
With off-the-shelf depth datasets, we need to generate json annotaions in compatible with this dataset, which is organized by:
```
dict(
	'files':list(
		dict(
			'rgb': 'data/kitti_demo/rgb/xxx.png',
			'depth': 'data/kitti_demo/depth/xxx.png',
			'depth_scale': 1000.0 # the depth scale of gt depth img.
			'cam_in': [fx, fy, cx, cy],
		),

		dict(
			...
		),

		...
	)
)
```
To generate such annotations, please refer to the "Inference" section.

### configs
In ```mono/configs``` we provide different config setups. 

Intrinsics of the canonical camera is set bellow: 
```
    canonical_space = dict(
        img_size=(512, 960),
        focal_length=1000.0,
    ),
```
where cx and cy is set to be half of the image size.

Inference settings are defined as
```
    depth_range=(0, 1),
    depth_normalize=(0.3, 150),
    crop_size = (512, 1088),
```
where the images will be first resized as the ```crop_size``` and then fed into the model.

## ✈️ Training
Please refer to [training/README.md](./training/README.md).
Now we provide complete json files for KITTI fine-tuning.

## ✈️ Inference
### News: Improved ONNX support with dynamic shapes (Feature owned by [@xenova](https://github.com/xenova). Appreciate for this outstanding contribution 🚩🚩🚩)

Now the onnx supports are availble for all three models with varying shapes. Refer to [issue117](https://github.com/YvanYin/Metric3D/issues/117) for more details.

### Improved ONNX Checkpoints Available now 
|      |       Encoder       |      Decoder      |                                               Link                                                |
|:----:|:-------------------:|:-----------------:|:-------------------------------------------------------------------------------------------------:|
| v2-S-ONNX | DINO2reg-ViT-Small  |    RAFT-4iter     | [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-small) |
| v2-L-ONNX | DINO2reg-ViT-Large  |    RAFT-8iter     | [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-large) |
| v2-g-ONNX | DINO2reg-ViT-giant2 |    RAFT-8iter     | [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-giant2) |

One additional [reminder](https://github.com/YvanYin/Metric3D/issues/143#issue-2444506808) for using these onnx models is reported by @norbertlink.

### News: Pytorch Hub is supported
Now you can use Metric3D via Pytorch Hub with just few lines of code:
```python
import torch
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
pred_depth, confidence, output_dict = model.inference({'input': rgb})
pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
```
Supported models: `metric3d_convnext_tiny`, `metric3d_convnext_large`, `metric3d_vit_small`, `metric3d_vit_large`, `metric3d_vit_giant2`.

We also provided a minimal working example in [hubconf.py](https://github.com/YvanYin/Metric3D/blob/main/hubconf.py#L145), which hopefully makes everything clearer.

### News: ONNX Exportation and Inference are supported

We also provided a flexible working example in [metric3d_onnx_export.py](./onnx/metric3d_onnx_export.py) to export the Pytorch Hub model to ONNX format. We could test with the following commands:

```bash
# Export the model to ONNX model
python3 onnx/metric_3d_onnx_export.py metric3d_vit_small # metric3d_vit_large/metric3d_convnext_large

# Test the inference of the ONNX model
python3 onnx/test_onnx.py metric3d_vit_small.onnx
```

[ros2_vision_inference](https://github.com/Owen-Liuyuxuan/ros2_vision_inference) provides a Python example, showcasing a pipeline from image to point clouds and integrated into ROS2 systems.

### Download Checkpoint
|      |       Encoder       |      Decoder      |                                               Link                                                |
|:----:|:-------------------:|:-----------------:|:-------------------------------------------------------------------------------------------------:|
| v1-T |    ConvNeXt-Tiny    | Hourglass-Decoder | [Download 🤗](https://huggingface.co/JUGGHM/Metric3D/blob/main/convtiny_hourglass_v1.pth)        |
| v1-L |   ConvNeXt-Large    | Hourglass-Decoder | [Download](https://drive.google.com/file/d/1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN/view?usp=drive_link) |
| v2-S | DINO2reg-ViT-Small  |    RAFT-4iter     | [Download](https://drive.google.com/file/d/1YfmvXwpWmhLg3jSxnhT7LvY0yawlXcr_/view?usp=drive_link) |
| v2-L | DINO2reg-ViT-Large  |    RAFT-8iter     | [Download](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link) |
| v2-g | DINO2reg-ViT-giant2 |    RAFT-8iter     | [Download 🤗](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_giant2_800k.pth) |

### Dataset Mode
1. put the trained ckpt file ```model.pth``` in ```weight/```.
2. generate data annotation by following the code ```data/gene_annos_kitti_demo.py```, which includes 'rgb', (optional) 'intrinsic', (optional) 'depth', (optional) 'depth_scale'.
3. change the 'test_data_path' in ```test_*.sh``` to the ```*.json``` path. 
4. run ```source test_kitti.sh``` or ```source test_nyu.sh```.

### In-the-Wild Mode
1. put the trained ckpt file ```model.pth``` in ```weight/```.
2. change the 'test_data_path' in ```test.sh``` to the image folder path. 
3. run ```source test_vit.sh``` for transformers and ```source test.sh``` for convnets.
As no intrinsics are provided, we provided by default 9 settings of focal length.

### Metric3D and Droid-Slam
If you are interested in combining metric3D and monocular visual slam system to achieve the metric slam, you can refer to this [repo](https://github.com/Jianxff/droid_metric).

## ❓ Q & A
### Q1: Why depth maps look good but pointclouds are distorted?
Because the focal length is not properly set! Please find a proper focal length by modifying codes [here](mono/utils/do_test.py#309) yourself.  

### Q2: Why the point clouds are too slow to be generated?
Because the images are too large! Use smaller ones instead. 

### Q3: Why predicted depth maps are not satisfactory?
First be sure all black padding regions at image boundaries are cropped out. Then please try again.
Besides, metric 3D is not almighty. Some objects (chandeliers, drones...) / camera views (aerial view, bev...) do not occur frequently in the training datasets. We will going deeper into this and release more powerful solutions.

## 📧 Citation
If you use this toolbox in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@misc{Metric3D,
  author =       {Yin, Wei and Hu, Mu},
  title =        {OpenMetric3D: An Open Toolbox for Monocular Depth Estimation},
  howpublished = {\url{https://github.com/YvanYin/Metric3D}},
  year =         {2024}
}
```
<!-- ```
@article{hu2024metric3dv2,
  title={Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation},
  author={Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
  journal={arXiv preprint arXiv:2404.15506},
  year={2024}
}
``` -->
Also please cite our papers if this help your research.
```
@article{hu2024metric3dv2,
  title={Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation},
  author={Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
```
@article{yin2023metric,
  title={Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image},
  author={Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen},
  booktitle={ICCV},
  year={2023}
}
```

## License and Contact

The *Metric 3D* code is under a 2-clause BSD License. For further commercial inquiries, please contact Dr. Wei Yin  [yvanwy@outlook.com] and Mr. Mu Hu [mhuam@connect.ust.hk].
