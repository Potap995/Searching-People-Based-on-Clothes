import threading
import cv2
import time
from queue import Queue
import PIL

import torch
import numpy as np
from profilehooks import profile
from .support_functions import resize_img

# TODO check exeptions while reading


class VideoStream:
    """ Класс для быстрого считывания видео.

    Запускается в отдельном потоке, с заданной queue_size.
    """
    def __init__(self, stream, visualizer=None, tracks_reader=None, queue_size=3, frame_size=None):
        """
        Ожидается что visualizer и tracks_reader будут оба либо None либо определены
        """

        self.stream = stream
        print(self.stream.get(cv2.CAP_PROP_FOURCC), "format")
        h = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        print(chr(h & 0xff) + chr((h >> 8) & 0xff) + chr((h >> 16) & 0xff) + chr((h >> 24) & 0xff))
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.last_read_num = 0

        self.stopped = False
        self.allreaded = False
        self.frame_size = frame_size

        self.queue_size = queue_size
        self.Q = Queue(maxsize=queue_size)

        self.thread = threading.Thread(target=self._run, args=())
        self.pause_cond = threading.Condition(threading.Lock())
        self.paused = False
        # self.size = None

        self.visualizer = visualizer
        self.tracks_reader = tracks_reader

        self.thread.start()

    # def start(self):
    #     self.thread.start()

    def _run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    # Проверка на нахождение в паузе
                    self.pause_cond.wait()

                if self.stopped:
                    # Завершение потока, вызывается только извне.
                    break
                if not self.allreaded:  # TODO split allreaded and exeptions(maybe)
                    if not self.Q.full():
                        self.prepare_frame()
                    else:
                        time.sleep(0.0001)
                else:
                    time.sleep(0.1)
        # Конец, завершение потока
        print("SROOOOOOOOOOP")
        self.stream.release()

    # @profile
    def prepare_frame(self):
        (grabbed, frame) = self.stream.read()
        if not grabbed:
            self.allreaded = True
        else:
            self.last_read_num += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            border = 1
            # TODO переписать тут
            if self.frame_size is not None:
                border = frame.shape[0] // self.frame_size[0]
                # frame = resize_img(frame, self.frame_size)
            if self.tracks_reader is not None:
                bboxes, colors = self.tracks_reader.get_tracks_at(self.last_read_num)
                frame = self.visualizer.draw_tracks(frame, bboxes, colors, border)
            self.Q.put(frame)

    def flush_queue(self):
        """ Очиста очереди

        Вызывать внутри паузы
        """
        self.Q = Queue(maxsize=self.queue_size)

    def reset_queue(self):
        """ Очищаем очередь

        Вызывается когда произошло изменения которые должны отразиться на кадре
        """
        self.pause()
        frame_num = self.last_read_num - self.Q.qsize()
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.flush_queue()
        self.pause()

    def set_frame_num(self, frame):
        """ Установка номера кадра для проигрывания

        Очищается отцередь
        """
        self.pause()
        self.last_read_num = int(frame)
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.last_read_num)
        self.flush_queue()
        self.allreaded = False
        self.resume()

    def get_next_frame(self):
        """ Получение следующего кадра

        Если в очереди ничего не находилось, то ожидание 5 * 0.005
        Возвращает True и сам кадр, если таковой был в очереди, иначе False
        """
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            print("warning!", threading.get_ident())
            time.sleep(0.005)
            tries += 1
        if self.Q.qsize() > 0 and not self.stopped:
            return True, self.Q.get()
        else:
            return False, []

    def set_frame_size(self, size: tuple):
        """ Установка нужного размера выходных кадров"""
        self.frame_size = size
        # Возможно надо будет чистить очередь или отстанавливать поток

    def get_fps(self):
        return self.fps

    def stop(self):
        """ Остановка потока для его завершения """
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()

    def pause(self):
        """ Постанока потока на паузу для изменения логики обработки """
        if self.paused:
            return
        self.paused = True
        self.pause_cond.acquire()

    def resume(self):
        """ Снятие потока с паузы """
        if not self.paused:
            return
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()

    def __del__(self):
        del self.stream
        del self.thread
        del self.Q



