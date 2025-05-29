import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy
import threading
import platform


SCALE_FACTOR = 2


# Get next worker's id
def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1


# Get previous worker's id
def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1


# A subprocess use to capture frames.
def capture(read_frame_list, Global, worker_num):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num, worker_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)

    # Release webcam
    video_capture.release()


# Many subprocess use to process frames.
def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    while not Global.is_exit:

        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:
                break

            time.sleep(0.01)

        # Delay to make the video look smoother
        time.sleep(Global.frame_delay)

        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]

        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num, worker_num)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        small_frame = cv2.resize(frame_process, (0, 0), fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR)
        rgb_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the frame of video, cost most time
        face_locations = face_recognition.face_locations(rgb_frame)
        gray_frame = cv2.cvtColor(rgb_frame , cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(gray_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(Global.known_face_encodings, face_encoding)
            top *= SCALE_FACTOR
            right *= SCALE_FACTOR
            bottom *= SCALE_FACTOR
            left *= SCALE_FACTOR

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = Global.known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num, worker_num)


def get_frames():
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    Global.known_face_encodings = []
    Global.known_face_names = []
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers (subprocess use to process frames)
    if cpu_count() > 2:
        worker_num = cpu_count() - 1  # 1 for capturing frames
    else:
        worker_num = 2

    p = []

    # Create a thread to capture frames
    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num,)))
    p[0].start()

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num,)))
        p[worker_id].start()

    yield Global

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / numpy.sum(fps_list)
            print("fps: %.2f" % fps)

            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.6
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            yield write_frame_list[prev_id(Global.write_num, worker_num)]

        time.sleep(0.01)
