import queue
import time
import threading

class TracksReader:

    def __init__(self, db):
        self.db = db

        self.conn = None
        self.cursor = None
        self.video_thread_id = None

        self.file = None

    def set_file(self, file):
        if self.conn is not None:
            self.conn = None
            self.cursor = None
            self.video_thread_id = None
        self.file = file

    def _get_trackd_form_db(self, frame_num):
        time1 = time.perf_counter()
        sql_select_bboxes = """\
                SELECT person_id, x_pos, y_pos, width, height
                FROM tracks_boxes
                WHERE file_id=? AND frame_num=?"""
        self.cursor.execute(sql_select_bboxes, [self.file.file_id, frame_num])
        ret = self.cursor.fetchall()
        time3 = time.perf_counter()
        # print("New:", time3 - time1)
        if time3 - time1 > 1 / 30:
            print("Execute: ", time3 - time1)
        return ret

    def get_tracks_at(self, frame_num, all_tracks=False):
        if self.conn is None:
            self.conn = self.db.get_connection()
            self.cursor = self.conn.cursor()
            self.video_thread_id = threading.get_ident()

        cur_thread_id = threading.get_ident()
        if cur_thread_id == self.video_thread_id:
            bboxes = self._get_trackd_form_db(frame_num)
        else:
            bboxes = self.db.get_tracks(self.file, frame_num)

        ret_bboxes = []
        for bbox in bboxes:
            if all_tracks or bbox[0] in self.file.visible_tracks:
                ret_bboxes.append(list(bbox))
        colors = []
        if self.file.tracks_list:
            for bbox in ret_bboxes:
                bbox_id = bbox[0]
                colors.append(self.file.tracks_list[bbox_id].color.tolist())
        return ret_bboxes, colors if len(colors) else None

    def close(self):
        if self.conn:
            self.conn.close()

        self.conn = None


