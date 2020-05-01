import datetime
import glob
import os

import click
import numpy as np
import tensorflow as tf
import tqdm

_ROOT = os.path.abspath(os.path.dirname(__file__))
TF_RECORDS = _ROOT + "/data/tf_records/"


def _parse_line(line):
	splits = line.strip().split(" ")
	l = splits[0],
	g = splits[1].split(":")[1]
	f = [split.split(":")[1] for split in splits[2:]]

	return int(g), np.array(f).astype(np.float32), int(l[0])


def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature


def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature


def serialize_example(q_id, feature, labels):
	feature = {
		'q_id': create_int_feature(q_id),
		'feature': create_float_feature(feature),
		'labels': create_int_feature(labels)
	}
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()


def create_tf_records(data_path, per_file_limit=50000):
	# import pdb;
	# pdb.set_trace()

	tf_path = TF_RECORDS + data_path.split("/")[-1].replace(".txt", "/")
	if not os.path.exists(tf_path):
		os.makedirs(tf_path)
	filename = tf_path + str(datetime.datetime.now().timestamp()) + ".tfrecord"
	tf_writer = tf.io.TFRecordWriter(filename)
	doc_counts = 0
	with open(data_path, 'r') as f:
		for line in tqdm.tqdm(f):
			g, f, l = _parse_line(line)
			example = serialize_example([g], f, [l])
			tf_writer.write(example)
			doc_counts += 1
			if doc_counts >= per_file_limit:
				tf_writer.write(example)
				doc_counts = 0
				tf_writer.close()
				filename = tf_path + str(datetime.datetime.now().timestamp()) + ".tfrecord"
				tf_writer = tf.io.TFRecordWriter(filename)
	tf_writer.close()


@click.command()
@click.option('--data-dir', type=str, default="/data/rank_data", show_default=True, help="training data path")
@click.option('--per-file-limit', type=int, default=50000, show_default=True, help="no of example per tfrecords")
def train(data_dir, per_file_limit):
	files = glob.glob(_ROOT + data_dir + "/*.txt")
	print("Creating TF Records...............")
	create_tf_records(files[0], per_file_limit)
	create_tf_records(files[1], per_file_limit)
	create_tf_records(files[2], per_file_limit)
	print("Tf Records Created................")


if __name__ == "__main__":
	train()
