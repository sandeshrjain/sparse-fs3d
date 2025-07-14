## [Datasets]

1. Download

    Download [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)(29GB), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)(12 GB), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)(16 MB)[labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)(5 MB)。Format the datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
    ```

    Additional KITTI Segmentation data could be used please refer to the KITTI 3D segmentation [source](https://www.cvlibs.net/datasets/kitti/eval_semantics.php)

    The generative samples were obtained via training based on aforementioned segmented 3D objects utilising the Point Cloud Diffusion [repository](https://github.com/luost26/diffusion-point-cloud/tree/main)   
   
2 . Pre-process KITTI datasets First

    ```
    cd sparse-fs3d/
    python pre_process_kitti.py --data_root your_path_to_kitti
    ```

    Now, we have datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
            |- velodyne_reduced (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
            |- velodyne_reduced (#7518 .bin)
        |- kitti_gt_database (# 19700 .bin)
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    ```

## [Training]
Note that the internal model imports the softpillars code under the models __init__.py file you may switch to other model adjustments via changing those imports (e.g., atten_pointpillars).
```
cd sparse-fs3d/
python train.py --data_root your_path_to_kitti
```
## [Evaluation]

```
cd sparse-fs3d/
python evaluate.py --ckpt pretrained/epoch_160.pth --data_root your_path_to_kitti 
```

## [Test]

```
cd sparse-fs3d/

# 1. infer and visualize point cloud detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path 
```



## Acknowledgments

This project builds upon the following open-source works:

- [PointPillars](https://github.com/zhulf0804/PointPillars) by @zhulf0804, licensed under the [MIT License](https://github.com/zhulf0804/PointPillars/blob/master/LICENSE)
- [Diffusion Point Cloud](https://github.com/luost26/diffusion-point-cloud) by @luost26, licensed under the [Apache License 2.0](https://github.com/luost26/diffusion-point-cloud/blob/main/LICENSE)

We have adapted and modified components from these projects. Please refer to the original repositories for full details.
