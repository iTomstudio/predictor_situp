# predictor_situp
基于Detectron2的仰卧起坐预测

[demo](http://player.bilibili.com/player.html?aid=370262057&bvid=BV14Z4y1x7mp&cid=178694480&page=1)

运行环境配置说明:[即Detectron2的环境配置](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

> 程序思路:请阅读 ReadMe 文件夹

issue：
1. AssertionError: Checkpoint model/keypoint_rcnn_R_101_FPN_3x.pkl not found!
  解决办法：可以参照 https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md 的demo，将官方demo的--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml 修改为运行 --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml ，detectron2 可以自行下载预训练模型。 即在 detectron2/demo 路径下，运行 python3 demo.py  --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml  --opts MODEL.DEVICE cpu
  
2.  AssertionError: Torch not compiled with CUDA enabled

    解决办法：运行时 添加参数： --opts MODEL.DEVICE cpu   ，使用方法 同上。


    

