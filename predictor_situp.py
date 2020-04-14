# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import math
import numpy as np
import matplotlib as mpl

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image,path):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

                extract_img = vis_output.img
                #画关键点
                keypoint_box = instances._fields["pred_keypoints"].numpy().tolist()
                img_name = path.split("/")[-1]
                if(len(keypoint_box)>0):
                    for idx, keypoint_list in enumerate(keypoint_box):
                        for idxx, keypoint in enumerate(keypoint_list):
                            pass
                        _ = self.write(extract_img,11,15,13,keypoint_list)
                        text_img = self.write(_,12,16,14,keypoint_list)
                    rgb = text_img[...,::-1]
                    cv2.imwrite("/home/dooncloud/GitHub/detectron2/output/self_"+img_name,rgb)

                # vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def calculate_angle(self, point_1,point_2,point_base):
        vector_a = [point_1[0]-point_base[0],point_1[1]-point_base[1]]
        vector_b = [point_2[0]-point_base[0],point_2[1]-point_base[1]]
        up = np.dot(vector_a,vector_b)
        a = np.linalg.norm(np.array(vector_a))
        b = np.linalg.norm(np.array(vector_b))
        down = a*b
        if down ==0:
            cos = 0.0
        else:
            cos = up/down
        if(abs(cos)>1):
            cos = 1
        return  math.degrees(math.acos(cos))

    def calculate_distance(self, point_1,point_2):
        vector = [point_1[0]-point_2[0],point_1[1]-point_2[1]]
        distance = np.linalg.norm(np.array(vector))
        return distance

    def where_point_write(self, n_list,keypoint_list):
        point_1 = keypoint_list[n_list[0]]
        point_2 = keypoint_list[n_list[1]]
        point_base = keypoint_list[n_list[2]]
        result = self.calculate_angle(point_1,point_2,point_base)
        x , y = point_base[0] , point_base[1]
        return result , x , y

    def write(self, img,need_list,keypoint_list):
        if len(need_list)>0:
            for i in need_list:
                result , x , y= self.where_point_write(i,keypoint_list)
                img = cv2.putText(
                    img, "%.2f"%result, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
        else:
            img = img
        return img

    def where_distance_write(self, n_list,keypoint_list):
        point_1 = keypoint_list[n_list[0]]
        point_2 = keypoint_list[n_list[1]]
        result = self.calculate_distance(point_1,point_2)
        x , y = point_2[0] , point_2[1]
        return result , x , y

    def write_distance(self, img,need_list,keypoint_list):
        if len(need_list)>0:
            for i in need_list:
                result , x , y= self.where_distance_write(i,keypoint_list)
                img = cv2.putText(
                    img, "%.2f"%result, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )
        else:
            img = img
        return img

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video, dictionary):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, dictionary):
            resulte = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )

            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                #判断框
                max_inform_keypoint = self.search_max_box_information(predictions)
                if(max_inform_keypoint != None):
                    #画框
                    bbox = max_inform_keypoint[0]
                    frame = cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),(0,255,0), 2)
                    # 画关键点
                    keypoint_list = max_inform_keypoint[1]
                    for i, keypoint in enumerate(keypoint_list):
                        circle_coord = (int(keypoint[0]),int(keypoint[1]))
                        frame = cv2.putText(
                            frame, str(i), circle_coord , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                        )
                    #画角度
                    frame = self.write(frame,dictionary["angle"],keypoint_list)
                    #画距离
                    frame = self.write_distance(frame,dictionary["distance"],keypoint_list)
                    #判断仰卧起坐
                    resulte = self.poll_situp(keypoint_list,dictionary)
                    #存结果
                    # save_json = self.save_resulte(keypoint_list,dictionary)
                    
                    vis_frame = frame[...,::-1]
                else:
                    vis_frame = frame[...,::-1]

                # vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            # vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            return {"vis_frame":vis_frame,"resulte": resulte,"max_inform_keypoint":max_inform_keypoint}

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame), dictionary)

    def poll_situp(self,keypoint_list,dictionary):
        ankle_angle_poll = self.angle_poll(dictionary["judge_ankle_angle"],keypoint_list,dictionary["require_ankle"])
        up_butt_angle_poll = self.angle_poll(dictionary["judge_butt_angle"],keypoint_list,dictionary["require_butt_up"])
        down_butt_angle_poll = self.angle_poll(dictionary["judge_butt_angle"],keypoint_list,dictionary["require_butt_down"])
        distance_ratio_poll = self.distance_ratio_poll(dictionary["judge_distance_ratio"],keypoint_list,dictionary["require_distance_ratio"])

        up_array = ankle_angle_poll + distance_ratio_poll+up_butt_angle_poll
        down_array = down_butt_angle_poll

        return [up_array,down_array]

    def save_resulte(self,keypoint_list,dictionary):
        ankle_num_list = self.calculate_save_angle(dictionary["judge_ankle_angle"],keypoint_list)
        
        log_f = open("digital","a+")
        print(ankle_num_list,file = log_f)
        log_f.close()

        butt_num_list = self.calculate_save_angle(dictionary["judge_butt_angle"],keypoint_list)

        distance_num_list = self.distance_poll(dictionary["judge_distance"],keypoint_list,dictionary["require_distance_ratio"])


    def calculate_save_angle(self,angle_list,keypoint_list):        
        resulte = []
        for  i in (angle_list):
            point_1 = keypoint_list[i[0]]
            point_2 = keypoint_list[i[1]]
            point_base = keypoint_list[i[2]]
            angle_result = self.calculate_angle(point_1,point_2,point_base)        
            resulte.append(angle_result)
        return resulte

    def angle_poll(self,angle_list,keypoint_list,requirement):
        poll = []
        resulte = self.calculate_save_angle(angle_list,keypoint_list)
        for idx, per_resulte in enumerate(resulte):
            if "<" is requirement["need"]:
                if per_resulte < requirement["angle"][idx]:
                    poll.append(1)
                else:
                    poll.append(0)
            elif ">" is requirement["need"]:
                if per_resulte > requirement["angle"][idx]:
                    poll.append(1)
                else:
                    poll.append(0)
            elif "=" is requirement["need"]:
                if per_resulte == requirement["angle"][idx]:
                    poll.append(1)
                else:
                    poll.append(0)
            else:
                raise Exception("calculate_dictionary  请输入正确判断符号")
        return poll

    def calculate_save_distance_ratio(self,distance_list,keypoint_list):        
        resulte = []
        for  i in (distance_list):
            point_1_1 = keypoint_list[i[0]]
            point_1_2 = keypoint_list[i[1]]
            point_2_1 = keypoint_list[i[2]]
            point_2_2 = keypoint_list[i[3]]

            up_result = self.calculate_distance(point_1_1,point_1_2)   
            down_resulte = self.calculate_distance(point_2_1,point_2_2)   
            ratio = up_result/down_resulte
            resulte.append(ratio)
        return resulte

    def distance_ratio_poll(self,distance_list,keypoint_list,requirement):
        poll = []
        resulte = self.calculate_save_distance_ratio(distance_list,keypoint_list)
        print(resulte)
        for idx, per_resulte in enumerate(resulte):
            if "<" is requirement["need"]:
                if per_resulte < requirement["distance"][idx]:
                    poll.append(1)
                else:
                    poll.append(0)
            elif ">" is requirement["need"]:
                if per_resulte > requirement["distance"][idx]:
                    poll.append(1)
                else:
                    poll.append(0)
            elif "=" is requirement["need"]:
                if per_resulte == requirement["distance"][idx]:
                    poll.append(1)
                else:
                    poll.append(0)
            else:
                raise Exception("calculate_dictionary  请输入正确判断符号")
        print(poll)
        return poll

    def search_max_box_information(self, predictions):
        keypoint_box_area = predictions._fields["pred_boxes"].area().numpy().tolist()
        keypoint_box_coordinate = predictions._fields["pred_boxes"].tensor.numpy().tolist()
        keypoint_box = predictions._fields["pred_keypoints"].numpy().tolist()
        assert (len(keypoint_box_area) == len(keypoint_box_coordinate) and len(keypoint_box_coordinate) == len(keypoint_box)) is True , "search max box --error"
        if(len(keypoint_box_area) != 0 ):
            if len(keypoint_box_area) >1 :
                index = keypoint_box_area.index(max(keypoint_box_area))
                return [keypoint_box_coordinate[index],keypoint_box[index]]
            else:
                return [keypoint_box_coordinate[0],keypoint_box[0]]
        else:
            pass
    

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
