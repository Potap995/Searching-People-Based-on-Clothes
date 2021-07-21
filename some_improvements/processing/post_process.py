import sys
import time
import logging
import argparse
import sqlite3
import itertools
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.spatial import cKDTree, distance_matrix
import cv2
from operator import methodcaller, add
from threading import Event

from support_functions import normalize, hex_to_rgb, rgb_to_hex

from pprint import pprint

conf = {"byteslen": 8192,
        "percentile": 75,
        "vector_size": 2048,
        "sim_dist": 0.55}


# TODO написать обертку над cKDtree

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_id", help="id of video file",
        default=1)
    parser.add_argument(
        "--video", help="path to video file",
        default=1)
    parser.add_argument(
        "--in_tracks", help="Path to tracks input .txt file",
        default=None)
    parser.add_argument(
        "--in_vectors", help="Path to vectors input .txt file",
        default=None)
    return parser.parse_args()


class TrackTime:

    def __init__(self, id_, gap_len=10):
        self.id_ = id_
        self.gap_len = gap_len
        self.segments = []
        self.cur_start = -1
        self.cur_end = -1

    def add(self, frame_num):
        if self.cur_start < 0:
            self._start_new_segment(frame_num)
        else:
            if self.cur_end + self.gap_len >= frame_num:
                self.cur_end = frame_num
            else:
                self._write_last()
                self._start_new_segment(frame_num)

    def _write_last(self):
        if self.cur_start > 0:
            self.segments.append((self.cur_start, self.cur_end))
            self.cur_start = -1

    def _start_new_segment(self, frame_num):
        self.cur_start = frame_num
        self.cur_end = frame_num

    def get_segments(self):
        self._write_last()
        return self.segments


def creat_temp_table_bbox(db_conn, video_id):
    cursor = db_conn.cursor()
    sql_creat_temp_table = f"""\
        CREATE TEMPORARY TABLE tracks_boxes_temp_{video_id} (
            file_id INTEGER NOT NULL,
            frame_num INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            x_pos REAL NOT NULL, 
            y_pos REAL NOT NULL, 
            width REAL NOT NULL, 
            height REAL NOT NULL
        );"""

    cursor.execute(sql_creat_temp_table)
    db_conn.commit()


def creat_temp_table_centers(db_conn, video_id):
    cursor = db_conn.cursor()
    sql_creat_temp_table = f"""\
            CREATE TEMPORARY TABLE tracks_centers_temp_{video_id} (
                file_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                vector BLOB,
                radius REAL NOT NULL
            );"""

    cursor.execute(sql_creat_temp_table)
    db_conn.commit()


def creat_temp_table_times(db_conn, video_id):
    cursor = db_conn.cursor()
    sql_creat_temp_table = f"""\
            CREATE TEMPORARY TABLE tracks_times_temp_{video_id} (
                file_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL
            );"""

    cursor.execute(sql_creat_temp_table)
    db_conn.commit()


def creat_tem_table_clothes(db_conn, video_id):
    cursor = db_conn.cursor()
    sql_creat_temp_table = f"""\
            CREATE TEMPORARY TABLE tracks_clothes_temp_{video_id} (
                file_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                clothes INTEGER NOT NULL,
                color TEXT NOT NULL,
                score REAL NOT NULL
            );"""

    cursor.execute(sql_creat_temp_table)
    db_conn.commit()


def creat_temp_table_frames(db_conn, video_id):
    cursor = db_conn.cursor()
    sql_creat_temp_table = f"""\
            CREATE TEMPORARY TABLE tracks_frames_temp_{video_id} (
                file_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                average_frame INTEGER NOT NULL,
                frames_num INTEGER NOT NULL
            );"""

    cursor.execute(sql_creat_temp_table)
    db_conn.commit()


def creat_temp_tables(db_conn, video_id):
    creat_temp_table_bbox(db_conn, video_id)
    creat_temp_table_centers(db_conn, video_id)
    creat_temp_table_times(db_conn, video_id)
    creat_tem_table_clothes(db_conn, video_id)
    creat_temp_table_frames(db_conn, video_id)


def get_vide_size(path):
    stream = cv2.VideoCapture(path)
    size = (stream.get(cv2.CAP_PROP_FRAME_WIDTH), stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream.release()
    return size


def read_all_bboxes(in_tracks_file):
    # ids_counter - количество кадров, на которых появлялся человек
    # boxes_max_area_frame - номер кадра с максимальной площадью bbox'a
    counter = 0
    ids_counter = Counter()
    boxes_max_area = Counter()
    boxes_max_area_frame = Counter()
    for line in tqdm(in_tracks_file):
        counter += 1

        line = line.rstrip("\n")
        line = line.split(",")

        frame_num = int(line[0])
        id_ = int(line[1])
        # x_pos, y_pos = line[2], line[3]
        width, height = line[4], line[5]

        box_area = (float(width) * float(height))
        if box_area > boxes_max_area[id_]:
            boxes_max_area[id_] = box_area
            boxes_max_area_frame[id_] = frame_num
        ids_counter[id_] += 1

    return counter, ids_counter, boxes_max_area_frame


def read_all_vectors(in_vectors_file, ids_counter, items_count, db_conn, video_id):
    ids_vectors = dict()
    for id_ in ids_counter:
        ids_vectors[id_] = np.zeros(conf["vector_size"], dtype=np.float32)

    for i in tqdm(range(items_count)):

        # TODO возможно тут надо иначе считывать вектора
        ind = int.from_bytes(in_vectors_file.read(4), byteorder='big')
        arr_s = in_vectors_file.read(conf["byteslen"])
        arr_new = np.frombuffer(arr_s, dtype=np.float32)
        if ind in ids_vectors:
            ids_vectors[ind] += arr_new / ids_counter[ind]  # тут может набегать ошибка
        else:
            print(ind)
            print("ID вектора нету среди ID")

    for id_ in ids_vectors:
        ids_vectors[id_] = normalize(ids_vectors[id_])

    return ids_vectors


def read_clothes(in_clothes_file, ids_counter):
    ids_clothes = dict()
    for id_ in ids_counter:
        ids_clothes[id_] = dict()

    for line in in_clothes_file:
        id_, clothes_all = line.split("|")
        clothes_all = clothes_all.strip()
        if clothes_all:
            id_ = int(id_)
            clothes_all = list(map(methodcaller("split", ":"), clothes_all.split(";")))  # -> [["class", "color"], ....]
            for clothes in clothes_all:
                clothes_class = int(clothes[0])
                color = hex_to_rgb(clothes[1])
                score = float(clothes[2])
                hist_tmp = clothes[3].translate({ord(i): None for i in '[]'})
                hist = np.array(list(map(float, hist_tmp.split(','))), dtype=float)
                if clothes_class not in ids_clothes[id_]:
                    ids_clothes[id_][clothes_class] = {'color': color, "count": 1, 'score': score, 'hist': hist}
                else:
                    # надеюсь тут не будет слишком больших чисел   10 ** 9 // 255 = 3921568
                    color = tuple(map(add, ids_clothes[id_][clothes_class]["color"], color))
                    counter_ = ids_clothes[id_][clothes_class]["count"] + 1
                    score = ids_clothes[id_][clothes_class]["score"] + score
                    hist = ids_clothes[id_][clothes_class]["hist"] + hist
                    ids_clothes[id_][clothes_class]["color"] = color
                    ids_clothes[id_][clothes_class]["count"] = counter_
                    ids_clothes[id_][clothes_class]["score"] = score
                    ids_clothes[id_][clothes_class]["hist"] = hist

    for id_ in ids_clothes:
        for clothes_class in ids_clothes[id_]:
            count_ = ids_clothes[id_][clothes_class]["count"]
            color = tuple(int(c / count_) for c in ids_clothes[id_][clothes_class]["color"])
            ids_clothes[id_][clothes_class]["color"] = rgb_to_hex(color)
            ids_clothes[id_][clothes_class]["score"] /= count_
            ids_clothes[id_][clothes_class]["hist"] /= count_
    return ids_clothes


def count_distances(in_vectors_file, ids_counter, ids_centers, items_count):
    ids_dists = dict()
    for id_ in ids_counter:
        ids_dists[id_] = []

    for _ in tqdm(range(items_count)):
        ind = int.from_bytes(in_vectors_file.read(4), byteorder='big')
        arr_s = in_vectors_file.read(conf["byteslen"])
        arr_new = np.frombuffer(arr_s, dtype=np.float32)
        if ind in ids_dists:
            ids_dists[ind].append(np.linalg.norm(ids_centers[ind] - arr_new))
        else:
            print(ind)
            print("ID вектора нету среди ID")

    return ids_dists


def count_ids_radius(ids_centers, ids_dists, percentile):
    ids_radius = dict()
    for id_ in ids_centers:
        ids_radius[id_] = np.percentile(np.asarray(ids_dists[id_]), percentile)
    return ids_radius


def get_data_for_tree(ids_centers, ids_radius):
    vectors = np.zeros((len(ids_centers), conf["vector_size"]))
    metadata = [None] * len(ids_centers)

    for i, id_ in enumerate(ids_centers):
        vectors[i] = ids_centers[id_]  # центры, вектор fast_reid
        metadata[i] = (id_, ids_radius[id_])  # (id трека, радиус)

    return vectors, metadata


def count_tree_accuracy(in_vectors_file, tree, metadata, items_count):
    count1 = 0
    count3 = 0
    count5 = 0
    count10 = 0
    for _ in tqdm(range(items_count)):
        ind = int.from_bytes(in_vectors_file.read(4), byteorder='big')
        arr_s = in_vectors_file.read(conf["byteslen"])
        arr_new = np.frombuffer(arr_s, dtype=np.float32)
        tracks_in_video = len(metadata[0])
        ds, inds = tree.query(arr_new, min(10, tracks_in_video))
        ids = []
        for i in inds:
            ids.append(metadata[i][0])

        if ind in ids[:1]:
            count1 += 1

        if ind in ids[:3]:
            count3 += 1

        if ind in ids[:5]:
            count5 += 1

        if ind in ids[:10]:
            count10 += 1

    return count1 / items_count, count3 / items_count, count5 / items_count, count10 / items_count


def save_centers(vectors, metadata, ids_new, db_conn, video_id):
    sql_insert_vectors = f"""\
            INSERT INTO tracks_centers_temp_{video_id} 
            (file_id, person_id, vector, radius) 
            VALUES ({video_id},?,?,?);"""

    cursor = db_conn.cursor()
    for i in tqdm(range(len(vectors))):
        vector_bytes = vectors[i].astype(np.float32).tobytes()
        radius = metadata[i][1]
        id_ = ids_new[metadata[i][0]]
        cursor.execute(sql_insert_vectors, (id_, vector_bytes, radius))

    db_conn.commit()
    cursor.close()


def save_bboxes(in_tracks_file, ids_new, size_from, db_conn, vide_id):
    sql_insert_tracks = f"""\
        INSERT INTO tracks_boxes_temp_{vide_id} 
        (file_id, frame_num, person_id, x_pos, y_pos, width, height) 
        VALUES ({vide_id},?,?,?,?,?,?);"""

    sql_insert_tracks_times = f"""\
        INSERT INTO tracks_times_temp_{vide_id} 
        (file_id, person_id, start, end) 
        VALUES ({vide_id},?,?,?);
    """

    cursor = db_conn.cursor()

    ids_segments = dict()
    for id_ in ids_new.values():
        ids_segments[id_] = TrackTime(id_)

    for line in tqdm(in_tracks_file):
        line = line.rstrip("\n")
        line = list(map(float, line.split(",")))
        line = line[:6]
        line[2] /= size_from[0]
        line[3] /= size_from[1]
        line[4] /= size_from[0]
        line[5] /= size_from[1]
        id_ = ids_new[int(line[1])]
        frame_num = int(line[0])
        line[0], line[1] = frame_num, id_
        ids_segments[id_].add(frame_num)
        cursor.execute(sql_insert_tracks, line)

    db_conn.commit()

    for id_ in ids_segments:
        for segment in ids_segments[id_].get_segments():
            cursor.execute(sql_insert_tracks_times, (id_, segment[0], segment[1]))

    db_conn.commit()
    cursor.close()


def save_clothes(clothes, ids_new, db_conn, video_id):
    sql_insert_vectors = f"""\
            INSERT INTO tracks_clothes_temp_{video_id} 
            (file_id, person_id, clothes, color, score) 
            VALUES ({video_id},?,?,?,?);"""

    cursor = db_conn.cursor()
    for id_ in clothes:
        for clothes_class in clothes[id_]:
            color = clothes[id_][clothes_class]["color"]
            id_ins = ids_new[id_]
            score = clothes[id_][clothes_class]["score"]
            # hist = clothes[id_][clothes_class]["hist"]
            cursor.execute(sql_insert_vectors, (id_ins, clothes_class, color, score))

    db_conn.commit()
    cursor.close()


def save_frames(frames_num_all, avg_frame_all, ids_new, db_conn, video_id):
    sql_insert_frames = f"""\
                INSERT INTO tracks_frames_temp_{video_id} 
                (file_id, person_id, average_frame, frames_num) 
                VALUES ({video_id},?,?,?);"""

    cursor = db_conn.cursor()

    for person_id in frames_num_all.keys():
        avg_frame = avg_frame_all[person_id]
        frames_num = frames_num_all[person_id]
        id_ = ids_new[person_id]

        cursor.execute(sql_insert_frames, (id_, avg_frame, frames_num))

    db_conn.commit()
    cursor.close()


def insert_into_main_db(db_conn, vide_id):
    sql_insert_bboxes = f"""\
    INSERT INTO tracks_boxes SELECT * FROM tracks_boxes_temp_{vide_id} 
    ;"""

    sql_insert_centers = f"""\
    INSERT INTO tracks_centers SELECT * FROM tracks_centers_temp_{vide_id} 
    ;"""

    sql_insert_times = f"""\
    INSERT INTO tracks_times SELECT * FROM tracks_times_temp_{vide_id} 
    ;"""

    sql_insert_clothes = f"""\
        INSERT INTO tracks_clothes SELECT * FROM tracks_clothes_temp_{vide_id} 
        ;"""

    sql_insert_frames = f"""\
        INSERT INTO tracks_frames SELECT * FROM tracks_frames_temp_{vide_id} 
        ;"""

    cursor = db_conn.cursor()

    cursor.execute(sql_insert_bboxes)
    cursor.execute(sql_insert_centers)
    cursor.execute(sql_insert_times)
    cursor.execute(sql_insert_clothes)
    cursor.execute(sql_insert_frames)

    db_conn.commit()
    cursor.close()


def get_db_files_ids(cursor):
    sql_select = """\
    SELECT file_id, processed FROM files
    ;"""
    cursor.execute(sql_select)
    rows = cursor.fetchall()
    # file_ids = list(itertools.chain.from_iterable(rows))
    return rows


def load_vectors_from_bd(cursor, file_id):
    sql_select = """\
    SELECT person_id, vector, radius 
    FROM tracks_centers 
    WHERE file_id=? 
    ;"""
    cursor.execute(sql_select, (file_id,))
    rows = cursor.fetchall()
    vectors = np.zeros((len(rows), conf["vector_size"]), dtype=np.float32)
    metadata = [None] * len(rows)
    for i, row in enumerate(rows):
        vectors[i] = np.frombuffer(row[1], dtype=np.float32)
        metadata[i] = (row[0], row[2])

    return vectors, metadata


def get_db_last_person_id(cursor):
    sql_select = """\
    SELECT MAX(person_id) FROM tracks_centers
    ;"""

    cursor.execute(sql_select)
    max_id = cursor.fetchone()[0]
    if max_id is None:
        max_id = 1
    return max_id


def compute_affinity_metric(first_sphere, second_sphere, dist):
    # TODO проработать метод compute_affinity_metric
    return first_sphere[1] + second_sphere[1] - dist


def add_nearest(tree, tree_data, cur_data, ids_metrics):
    tracks_in_video = len(tree_data[0])
    for track_ind in range(len(cur_data[0])):
        cur_vector, cur_radius = cur_data[0][track_ind], cur_data[1][track_ind][1]
        cur_sphere = (cur_vector, cur_radius)

        distances, ret_indexes = tree.query(cur_vector, min(3, tracks_in_video))
        cur_id = cur_data[1][track_ind][0]
        for i in range(len(ret_indexes)):
            i_vector = tree_data[0][ret_indexes[i]]
            i_radius = tree_data[1][ret_indexes[i]][1]
            i_id = tree_data[1][ret_indexes[i]][0]

            i_sphere = (i_vector, i_radius)
            metric = compute_affinity_metric(cur_sphere, i_sphere, distances[i])

            ids_metrics[cur_id].append((i_id, metric, distances[i]))


def choose_id(choices):
    # пока это выбор наименьшего расстояния
    min_dist = 10
    min_i = -1
    for i in range(len(choices)):
        if min_dist > choices[i][2]:
            min_dist = choices[i][2]
            min_i = i

    if min_dist > conf["sim_dist"]:  # 0.75
        return -1
    else:
        print("Совпадение")
        return choices[min_i][0]


def make_mapping(ids_metrics, last_id):
    ids_new = dict.fromkeys(ids_metrics.keys(), -1)
    for id_ in ids_metrics:
        new_id = choose_id(ids_metrics[id_])
        if new_id == -1:
            new_id = last_id + 1
            last_id += 1
        ids_new[id_] = new_id
    return ids_new


def map_ids(data, bd_conn, video_id):
    cursor = bd_conn.cursor()
    file_ids = get_db_files_ids(cursor)
    print(file_ids)
    last_person_id = get_db_last_person_id(cursor)
    print("last_person_id: ", last_person_id)

    ids_metrics = dict()
    for i in range(len(data[1])):
        ids_metrics[data[1][i][0]] = []

    for file_id, processed_flag in file_ids:
        if file_id == video_id or not processed_flag:
            continue
        saved_data = load_vectors_from_bd(cursor, file_id)
        tree = cKDTree(saved_data[0])
        add_nearest(tree, saved_data, data, ids_metrics)

    ids_new = make_mapping(ids_metrics, last_person_id)

    return ids_new


class PostProcess:

    # TODO перенести методы в класс что бы хорошо обрабатывать проценты. Или как то иначе
    def __init__(self, args, stop_event=Event()):
        self.stop_event = stop_event
        self.args = args
        self._percent = 0
        self._frequency = 100

    def run(self):
        conn = sqlite3.connect(self.args["db_file"])
        in_tracks_file = open(self.args["in_tracks"], 'r')
        in_vectors_file = open(self.args["in_vectors"], 'rb')
        in_clothes_file = open(self.args["in_clothes"], 'r')
        video_id = self.args["video_id"]
        size = get_vide_size(self.args['video'])

        creat_temp_tables(conn, video_id)

        items_count, ids_counter, boxes_max_area_frame = read_all_bboxes(in_tracks_file)

        print("bbox were read")
        if self.stop_event.is_set():
            return

        self._percent = int(0.05 * self._frequency)

        ids_centers = read_all_vectors(in_vectors_file, ids_counter, items_count, conn, video_id)

        if self.stop_event.is_set():
            return
        self._percent = int(0.23 * self._frequency)
        print("vectors were read")

        in_vectors_file.seek(0)
        ids_dists = count_distances(in_vectors_file, ids_counter, ids_centers, items_count)
        ids_radius = count_ids_radius(ids_centers, ids_dists, conf["percentile"])
        del ids_dists

        if self.stop_event.is_set():
            return
        self._percent = int(0.46 * self._frequency)
        print("distances were calculated")

        in_vectors_file.seek(0)
        vectors, metadata = get_data_for_tree(ids_centers, ids_radius)
        clothes = read_clothes(in_clothes_file, ids_counter)

        del ids_centers, ids_radius

        if self.stop_event.is_set():
            return
        tree = cKDTree(vectors)
        print(count_tree_accuracy(in_vectors_file, tree, metadata, items_count))

        self._percent = int(0.72 * self._frequency)
        print("tree has been build")
        ids_new = map_ids((vectors, metadata), conn, video_id)
        in_tracks_file.seek(0)
        if self.stop_event.is_set():
            return
        save_bboxes(in_tracks_file, ids_new, size, conn, video_id)
        save_centers(vectors, metadata, ids_new, conn, video_id)
        save_clothes(clothes, ids_new, conn, video_id)
        # in_tracks_file.seek(0)
        save_frames(ids_counter, boxes_max_area_frame, ids_new, conn, video_id)

        del ids_counter, boxes_max_area_frame

        print("info saved in temp")

        if self.stop_event.is_set():
            return
        self._percent = int(0.88 * self._frequency)

        insert_into_main_db(conn, video_id)
        # res = distance_matrix(vectors, vectors)
        # print(res.shape[0])
        # print(np.sum(res < 0.5) - res.shape[0])
        self._percent = int(1 * self._frequency)
        print("end")

        in_tracks_file.close()
        in_vectors_file.close()
        conn.close()
        return

    @property
    def percent(self):
        return self._percent


def run_post_process(args):
    process = PostProcess(args)
    return process.run()


def main(args):
    args = vars(args)
    args['video_id'] = 111
    args['video'] = "./../Tracks/data/vide.mp4"
    args['in_tracks'] = "output/video_out_tracks.txt"
    args['in_vectors'] = "output/video_out_vectors.txt"

    run_post_process(args)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
