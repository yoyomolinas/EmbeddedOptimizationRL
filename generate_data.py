import sys
sys.path.append("../../")

from absl import app, flags, logging
from absl.flags import FLAGS

import numpy as np
import copy
import cv2
import time

from Deployment.buffer import VideoCapture
from Deployment.deep_sort import NearestNeighborDistanceMetric
from Deployment.deep_sort import Tracker
from Deployment.pose import RectangleTarget
from Deployment.database.jsondb import JsonDB

flags.DEFINE_string('video', None, 'video to analyze, empty for camera')
flags.DEFINE_boolean('visualize', False, 'vidualize detections and tracks')
flags.DEFINE_boolean('mac', False, 'set true if running on Mac OS')
flags.DEFINE_string('save_to', None, 'file to save data into')
flags.DEFINE_integer('print_every', 1, 'prints every ith second')

def visualize_tracks(frame = None, detection_encodings= None, tracker = None):
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        t, l, b, r = track.to_tlbr().astype(np.int32)
        cv2.rectangle(frame, (l, t), (r, b), track.color, 6)
        looking = track.look_track.is_confirmed()
        if looking:
            frame[t:b, l:r][:, :, 0] = 255

    # Remove comment from the following line to print age-gender and display landmarks
    # frame = visualize(frame=frame, detection_encodings=detection_encodings, tracking = True)
    return frame

def visualize(frame = None, detection_encodings= None, tracking = False):
    frame = copy.deepcopy(frame)
    for detection in detection_encodings:
        try:
            t, l, b, r = detection.getBox(type = np.int32)
            if not tracking:
                cv2.rectangle(frame, (r, b), (l, t), (200, 100, 100), 2)
        except ValueError as e:
            logging.info(e)
            logging.info('Detection has no box. Weird..')
        try:
            gender = detection.getGender()
            age = detection.getAge()
            txt = gender + ' ' + str(age)
            cv2.putText(frame, txt, (int(l) + 2, int(t) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 120), 2)
        except AssertionError as e:
            logging.info(e)

        try:
            # Plot landmarks
            color = (255, 255, 0)
            marks = detection.getLandmarks()
            cv2.polylines(frame, np.int32([marks[:16]]), False, color)
            cv2.polylines(frame, np.int32([marks[17:22]]), False, color)
            cv2.polylines(frame, np.int32([marks[22:27]]), False, color)
            cv2.polylines(frame, np.int32([marks[27:31]]), False, color)
            cv2.polylines(frame, np.int32([marks[31:36]]), False, color)
            cv2.polylines(frame, np.int32([marks[36:42]]), True, color)
            cv2.polylines(frame, np.int32([marks[42:48]]), True, color)
            cv2.polylines(frame, np.int32([marks[60:]]), True, color)
            cv2.polylines(frame, np.int32([marks[48:60]]), True, color)
        except ValueError as e:
            logging.info(e)
    return frame

def main(_argv):
    shape = (720, 1280) # TODO Should match with gstreamer
    if FLAGS.video is not None:
        gst_pipeline = FLAGS.video
    elif FLAGS.mac:
        gst_pipeline = 0
    else:
        gst_pipeline = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

    vid = VideoCapture(pipeline=gst_pipeline, shape = shape)

    if FLAGS.mac:
        optimized = False
    else:
        optimized = True

    from Deployment.wrappers.component_model import ModelWrapper

    model = ModelWrapper(input_shape = shape, optimized=optimized) # Wrapped model with current build

    rectangle_target = RectangleTarget(width = 500, height = 1000, min_z_distance= 2000)
    metric = NearestNeighborDistanceMetric(metric='cosine', matching_threshold=0.25, budget=100) # Linear Assignment metric
    tracker = Tracker(metric=metric, frame_shape=shape, target=rectangle_target, n_init = 3)

    if FLAGS.save_to:
        db = JsonDB(FLAGS.save_to, dump_info={'video' : FLAGS.video})

    try:
        verbose_counter = 0
        while True:
            verbose_counter += 1
            tic_all = time.time()  # Time start for all
            try:
                frame = vid.read() # Read frame
            except BufferError as e:
                logging.info(e)
                if FLAGS.video:
                    break
                else:
                    continue

            # Detect and encode detections to be passed on to the tracker
            tic_inf = time.time() # Time start for inference
            encodings = model.do(image=frame, option = model.options.BoxLandmarkFeature)
            toc_inf = time.time() # Time finish for inference

            # Update tracker state
            tic_sort = time.time()  # Time start for Deep SORT
            tracker.predict()
            tracker.update(encodings)
            toc_sort = time.time()  # Time finish for Deep SORT

            # Further detections for rl build. Specifically detect age and gender
            tic_inf_second = time.time() # Time to detect age and gender
            encodings = model.do(option = model.options.AgeGender, encodings = encodings)
            toc_inf_second = time.time()

            if FLAGS.visualize:
                vis_frame = visualize_tracks(frame=frame, detection_encodings=encodings, tracker=tracker)
                vis_frame = vid.rgb2bgr(vis_frame)
                cv2.imshow('frm', vis_frame)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            if FLAGS.save_to:
                data = []
                for enc in encodings:
                    toc_duration = time.time()
                    enc.updateDuration(toc_duration - tic_all)
                    data.append(enc.to_dict(include_landmarks=False, include_feature=True))
                db.write(data)

            toc_all = time.time()  # Time finish for all

            if verbose_counter % FLAGS.print_every == 0: # log every 0 seconds
                verbose_counter = 0
                logging.info('\n---------------------------\nFrame %i\n'%vid.frame_count)
                logging.info('Total time:%.3f'%round(toc_all - tic_all, 3))
                logging.info('Inference time: %.3f'% round(toc_inf - tic_inf, 3))
                logging.info('Second inference time" %.3f'%round(toc_inf_second - tic_inf_second, 3))
                logging.info('Tracking time: %.3f'%round(toc_sort - tic_sort, 3))


    except KeyboardInterrupt as e:
        logging.info("Keyboard interrupt occured. ")

    finally:
        tic_dump = time.time() # Time start for dumping into file
        if FLAGS.save_to:
            # Lastly, dump data into file
            db.dump()
        toc_dump = time.time() # Time end for dumping into file
        logging.info('Dump time: %.3f'%round(toc_dump - tic_dump, 3))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
