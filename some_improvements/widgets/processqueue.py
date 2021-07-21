from multiprocessing import Process
from threading import Thread, Event
import concurrent.futures
from processing import RepeatedTimer, VideoProcess


class ProcessingQueue:

    def __init__(self, db):
        self._queue = []
        self._current = None
        self.db = db
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._stopped = False
        self._stop_event = Event()
        self._future = None

    def add_to_queue(self, file_widget):
        video_item = OneVideoProcess(file_widget, len(self._queue) + 1, self._stop_event, self.db.db_file_path)
        self._queue.append(video_item)

        self._run_next()

    def _run_next(self):
        if self._current is not None:
            return

        if not self._queue:
            return

        self._current = self._queue[0]
        self._current.update_queue_num(0)

        idx = 0
        while idx < len(self._queue) - 1:
            self._queue[idx] = self._queue[idx + 1]
            self._queue[idx].update_queue_num(idx + 1)
            idx += 1

        del self._queue[-1]
        if not self._stopped:
            self._future = self.executor.submit(self._current.run)
            self._future.add_done_callback(self._callback_processed)

    def _callback_processed(self, _future):
        cur_file = self._current.file
        if not self._stopped:
            self.set_processed(cur_file)
        self._current.update_processed()
        self._current = None
        self._run_next()

    def set_processed(self, file):
        file.set_processed()
        conn = self.db.get_connection()
        cursor = conn.cursor()
        sql_update_file = """\
        UPDATE files
        SET processed = 1
        WHERE file_id = ?
        ;"""
        cursor.execute(sql_update_file, (file.file_id,))
        conn.commit()
        conn.close()

    def stop(self):
        # Я не знаю как завершить процесс
        self._stopped = True
        self._stop_event.set()
        if self._future:
            self._future.cancel()
        self.executor.shutdown()


class OneVideoProcess:

    def __init__(self, file_widget, queue_num, stop_event, db_path):
        self._file_widget = file_widget
        self.file = file_widget.file

        self._timer = RepeatedTimer(1, self._update_percent_label)
        self._video_process = VideoProcess(self.file.name, self.file.path, self.file.file_id, stop_event, db_path)
        self._process = Thread(target=self._video_process.run, daemon=True)

        self._queue_num = -1
        self.update_queue_num(queue_num)

    def run(self):
        self._timer.start()
        self._process.start()
        self._process.join()
        self._update_percent_label()
        self._timer.stop()

    def _update_percent_label(self):
        state_text = f"{self._video_process.get_percent()}%"
        self._file_widget.set_processing_state(state_text)

    def update_queue_num(self, queue_num):
        self._queue_num = queue_num
        state_text = f"->({self._queue_num})->"
        self._file_widget.set_processing_state(state_text)

    def update_processed(self):
        self._file_widget.set_processing_state("Обработан")
