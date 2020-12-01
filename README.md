# predictor_situp
基于Detectron2的仰卧起坐预测

[demo](http://player.bilibili.com/player.html?aid=370262057&bvid=BV14Z4y1x7mp&cid=178694480&page=1)

运行环境配置说明:[即Detectron2的环境配置](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

> 程序思路:请阅读 ReadMe 文件夹

Run:   python3 demo_situp.py  --video-input cut.mp4     ( Linux )

issue：
1. AssertionError: Checkpoint model/keypoint_rcnn_R_101_FPN_3x.pkl not found!
  
    链接: https://pan.baidu.com/s/1CIjF-GHqCk6FGW-ABxEzYA 提取码: d5uf  下载放置：model/
  
2.  AssertionError: Torch not compiled with CUDA enabled

    解决办法：运行时 添加参数： --opts MODEL.DEVICE cpu  


    

