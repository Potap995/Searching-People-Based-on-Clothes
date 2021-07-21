import sqlite3
import time
import itertools
from collections import defaultdict
import os
from .creatDB import create_db
from pathlib import Path


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)

    return conn


class MyDB:

    def __init__(self, db_path):
        self.db_file_path = None
        self.conn = None
        self.set_db_file(db_path)

    def set_db_file(self, db_path):
        if self.conn is not None:
            self.conn.close()
        self.db_file_path = Path(db_path)
        self.conn = create_connection(db_path)
        self.conn.row_factory = sqlite3.Row

        # TODO написать проверку того что база данных валидная

    def get_name(self):
        return self.db_file_path.stem

    def get_connection(self):
        return create_connection(self.db_file_path)

    def get_files_list(self):
        cursor = self.conn.cursor()

        sql_request = "SELECT * FROM files;"
        cursor.execute(sql_request)

        rows = cursor.fetchall()

        return rows

    def add_file(self, file):
        cursor = self.conn.cursor()
        sql_insert = "INSERT INTO files (name, video_path, processed) VALUES (?,?,?);"
        data = [file.name, file.relative_path, file.processed]

        try:
            cursor.execute(sql_insert, data)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise FileExistsError(f"Файл с именем {file.name} уже существует")
        except Exception as e:
            print("Лол, а это что еще за ошибка в add_file")
        else:
            arr = cursor.lastrowid
            cursor.close()
            return arr
        return None

    def get_tracks_time(self, file):
        time1 = time.perf_counter()
        cursor = self.conn.cursor()
        sql_select = """\
        SELECT DISTINCT person_id, start, end FROM tracks_times WHERE file_id = ? 
        ;"""
        cursor.execute(sql_select, (file.file_id,))
        rows = cursor.fetchall()
        ids_segments = defaultdict(list)
        for row in rows:
            ids_segments[row[0]].append((row[1], row[2]))
        time2 = time.perf_counter()
        if time2-time1 > 0.3:
            print("Что то долгое время загрузки списка треков в видео")
        return ids_segments

    def get_tracks_info(self, file):
        time1 = time.perf_counter()
        cursor = self.conn.cursor()
        sql_select = """\
        SELECT DISTINCT person_id, average_frame, frames_num FROM tracks_frames WHERE file_id = ? 
        ;"""
        cursor.execute(sql_select, (file.file_id,))
        rows = cursor.fetchall()
        ids_info = dict()
        for row in rows:
            ids_info[row[0]] = {"mid": row[1], "num": row[2]}
        time2 = time.perf_counter()
        if time2-time1 > 0.3:
            print("Что то долгое время загрузки списка треков в видео")
        return ids_info

    def get_track_bbox_from_frame(self, track, frame_num):
        cursor = self.conn.cursor()
        sql_select_bbox = """\
                SELECT x_pos, y_pos, width, height
                FROM tracks_boxes
                WHERE file_id=? AND frame_num=? AND person_id=?"""
        cursor.execute(sql_select_bbox, [track.file.file_id, frame_num, track.track_id])
        ret = cursor.fetchall()
        return ret

    def get_all_vectors(self):
        cursor = self.conn.cursor()
        sql_select = """SELECT * FROM tracks_centers"""
        cursor.execute(sql_select)
        return cursor

    def get_clothes_names(self):
        cursor = self.conn.cursor()
        sql_select = """SELECT clothes_id, name FROM clothes_names"""
        cursor.execute(sql_select)
        return cursor.fetchall()

    def get_track_clothes(self, track):
        cursor = self.conn.cursor()
        sql_select = """SELECT ROWID, clothes, color, score FROM tracks_clothes WHERE file_id=? and person_id=?"""
        cursor.execute(sql_select, (track.file.file_id, track.track_id))
        return cursor.fetchall()

    def get_all_clothes(self):
        cursor = self.conn.cursor()
        sql_select = """SELECT * FROM tracks_clothes"""
        cursor.execute(sql_select)
        return cursor

    def get_tracks(self, file, frame_num):
        cursor = self.conn.cursor()
        sql_select_bboxes = """\
        SELECT person_id, x_pos, y_pos, width, height
        FROM tracks_boxes
        WHERE file_id=? AND frame_num=?"""
        cursor.execute(sql_select_bboxes, [file.file_id, frame_num])
        ret = cursor.fetchall()
        return ret

    def get_track_vector_on_file(self, track):
        cursor = self.conn.cursor()
        sql_select_vector = """\
                SELECT vector, radius
                FROM tracks_centers
                WHERE file_id=? AND person_id=?"""
        cursor.execute(sql_select_vector, [track.file.file_id, track.track_id])
        ret = cursor.fetchall()
        return ret

    def del_clothes(self, clothes):
        cursor = self.conn.cursor()
        sql_dell = "DELETE FROM tracks_clothes WHERE ROWID=?"
        for item_ in clothes:
            print(f"del db_id = {item_.db_id}")
            cursor.execute(sql_dell, (item_.db_id,))
        self.conn.commit()
        cursor.close()

    # def change_clothes(self, clothes: list, person):
    #     cursor = self.conn.cursor()
    #
    #     cursor.close()
    #     self.conn.commit()

    def add_clothes(self, clothes: list, person):
        cursor = self.conn.cursor()
        rowids = []
        sql_insert = "INSERT INTO tracks_clothes (file_id, person_id, clothes, score, color) VALUES (?,?,?,?,?);"
        for item_ in clothes:
            cursor.execute(sql_insert, (person.file.file_id, person.track_id,
                                        item_.clothes_id, item_.score, item_.color))
            rowids.append(cursor.lastrowid)
        cursor.close()
        self.conn.commit()
        return rowids

    def search_saved_img_info(self, info):
        cursor = self.conn.cursor()
        if info["out"]:
            sql_query = """
            SELECT img_id FROM mistakes_img_info_out
            WHERE img_path=?
            ;"""
            sql_data = (info["path"],)
        else:
            sql_query = """
            SELECT img_id FROM mistakes_img_info_in
            WHERE track_id=? and file_id=? and frame=?
            ;"""
            sql_data = (info["track_id"], info["file_id"], info["frame_from"])

        cursor.execute(sql_query, sql_data)
        cur_id = cursor.fetchone()
        cursor.close()
        if cur_id is not None:
            cur_id = int(cur_id[0])

        return cur_id

    def get_last_img_id(self):
        cursor = self.conn.cursor()
        sql_select = """
        SELECT value FROM info
        WHERE key = "last_save_mistake_id"
        ;"""

        cursor.execute(sql_select)
        max_id = cursor.fetchone()[0]
        cursor.close()

        if max_id is None:
            max_id = 0
        return int(max_id)

    def save_last_img_id(self, last_id):
        cursor = self.conn.cursor()
        sql_select = """
        UPDATE info SET value=? 
        WHERE key="last_save_mistake_id"
        ;"""

        cursor.execute(sql_select, (last_id,))
        self.conn.commit()
        cursor.close()

    def save_img_info(self, info, img_id):
        cursor = self.conn.cursor()
        if info["out"]:
            sql_query = """
            INSERT INTO mistakes_img_info_out (img_path, img_id)
            VALUES (?, ?)
            ;"""
            sql_data = (info["path"], img_id)
        else:
            sql_query = """
            INSERT INTO mistakes_img_info_in (track_id, file_id, frame, img_id)
            VALUES (?, ?, ?, ?)
            ;"""
            sql_data = (info["track_id"], info["file_id"], info["frame_from"], img_id)

        cursor.execute(sql_query, sql_data)
        self.conn.commit()
        cursor.close()

    def save_mistake(self, first_id, second_id, dist, mistake_type):
        cursor = self.conn.cursor()
        sql_add_mistake = """
        INSERT INTO mistakes (first , second , dist, type)
        VALUES (?, ?, ?, ?);"""

        cursor.execute(sql_add_mistake, (first_id, second_id, dist, mistake_type))
        self.conn.commit()
        cursor.close()

    def get_track_mistakes(self, track, search_img_id):
        cursor = self.conn.cursor()
        sql_query = """
        SELECT type from mistakes
        WHERE first=? and second in (
            SELECT img_id from mistakes_img_info_in
            WHERE track_id=?
        );"""

        cursor.execute(sql_query, (search_img_id, track.track_id))
        match_type = cursor.fetchone()
        cursor.close()
        if match_type is not None:
            match_type = int(match_type[0])

        return match_type

    def __del__(self):
        print("Close connection")
        if self.conn:
            self.conn.close()


