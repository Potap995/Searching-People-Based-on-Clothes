import threading
import cv2
import time
from queue import Queue
from Visualizer import Visualizer


class FileVideoStream:
    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue_size = queue_size
        self.path = path
        self.Q = Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self.update, args=())
        # self.thread.daemon = True
        self.allreaded = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.paused = False
        self.Visualizer = Visualizer.getInstance()
        self.Visualizer.setStream(self)
        self.frame_id = 0

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()

                if self.stopped:
                    break
                if self.allreaded:
                    continue
                if not self.Q.full():
                    (grabbed, frame) = self.stream.read()
                    self.frame_id += 1
                    if not grabbed:
                        self.allreaded = True

                    frame = self.Visualizer.drowTracks(frame, self.frame_id + 1)

                    self.Q.put(frame)
                else:
                    time.sleep(0.2)

        self.stream.release()

    def read(self):
        return self.Q.get()

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def flushQueue(self):
        self.Q = Queue(maxsize=self.queue_size)

    def setFrame(self, frame):
        self.pause()
        self.allreaded = False
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.flushQueue()
        self.frame_id = frame
        self.resume()


    def resetFrames(self):
        self.frame_id -= self.Q.qsize()
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        self.flushQueue()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()

    def pause(self):
        if self.paused:
            return
        self.paused = True
        self.pause_cond.acquire()

    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()
