import sys
sys.path.append(".")

from video_process import VideoProcess
from support_functions import RepeatedTimer

from multiprocessing import Process

def test():
    res = VideoProcess("ADL_1_1", "./data/video/ADL-Rundle-6_1.mp4", 1)


    def timer_print(obj, name=""):
        print(name, obj.get_percent())

    timer = RepeatedTimer(0.5, timer_print, res, "ADL_1_1")
    timer.start()

    procc = Process(target=res.run)
    procc.start()
    procc.join()

    timer_print(res, "ADL_1_1")
    timer.stop()

    print("END")

test()

