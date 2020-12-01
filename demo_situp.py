# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor_situp import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="model/keypoint_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def continue_frame_judge(store_list,self_cfg):
    up_poll = 0
    sit_poll = 0
    judge=[0,0]
    # print(store_list[-3:-1],store_list)
    if len(store_list) > self_cfg["judge_fps"]:
        start_num = -1-self_cfg["judge_fps"]
        #判断     
        for per_list in store_list[start_num:-1]:
            print(per_list)
            up_poll += sum(per_list[0]) 
            sit_poll += sum(per_list[1]) 

        if sit_poll > self_cfg["sit_poll"]:
            judge[1] =  1 
        if up_poll > self_cfg["up_poll"]:
            judge[0]= 1 
    else:
        judge = [0, 0]
    return judge

def continue_frame(self_cfg,resulte_storage,continue_frame_resulte_list):
    resulte_storage.append(continue_frame_resulte_list)
    resulte  = continue_frame_judge(resulte_storage,self_cfg)

    return resulte

def count_situp(statistic_state_resulte, cycle_state, count):
    if  cycle_state == 0:
        is_count = 0
        if is_count == 0:
            if statistic_state_resulte[0] == 1:
                count += 1
                cycle_state = 1
                is_count = 1
    if cycle_state == 1:
        if statistic_state_resulte[1] == 1:
            cycle_state = 0
                
    print(cycle_state)
    return  count, cycle_state
    

if __name__ == "__main__":
    situp_count = 0
    cycle_state = 1
    resulte_storage = []
    self_cfg = { "judge_fps":5,"up_poll":20,"sit_poll":6}
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        print(width)
        print(height)
        print(frames_per_second)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        basename = "vedio"

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mp4"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)

        calculate_dictionary = {
                                    "angle":[[11,15,13],[12,16,14], [6,14,12], [5,13,11]],
                                    "distance":[[1,15],[2,16]],
                                    "judge_ankle_angle":[[12,16,14],[11,15,13]],
                                    "judge_butt_angle":[ [6,14,12], [5,13,11] ],
                                    "judge_distance_ratio":[[16,14,14,2],[15,13,13,1]],
                                    "require_ankle":{"need":"<","angle":[100,90]},
                                    "require_butt_up":{"need":"<","angle":[110,110]},
                                    "require_butt_down":{"need":">","angle":[150,150]},
                                    "require_distance_ratio":{"need":">","distance":[0.5,0.6]}
                                    }
        # log_f = open("digital","w+")
        # log_f.close()

        for  ordinal_num, resulte_dictionary in enumerate(tqdm.tqdm(demo.run_on_video(video,calculate_dictionary), total=num_frames)):
            vis_frame = resulte_dictionary["vis_frame"]

            if(resulte_dictionary["max_inform_keypoint"]!= None):
                continue_frame_resulte_list = resulte_dictionary["resulte"]
                # print(continue_frame_resulte_list)
                statistic_state_resulte  =  continue_frame(self_cfg,resulte_storage,continue_frame_resulte_list)

                situp_count , cycle_state= count_situp(statistic_state_resulte, cycle_state, situp_count)
                
                #   显示 
                up_result = str(statistic_state_resulte[0])
                sit_result = str(statistic_state_resulte[1])
                vis_frame = vis_frame.astype('uint8')
                vis_frame = cv2.putText(
                    vis_frame, "up:"+up_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
                )
                vis_frame = cv2.putText(
                    vis_frame, "down:"+sit_result, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
                )
                vis_frame = cv2.putText(
                    vis_frame, "count:"+str(situp_count), (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
                )

            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
                # cv2.waitKey(0)
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
