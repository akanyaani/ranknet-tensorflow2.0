import tensorflow as tf

_READ_RECORD_BUFFER = 8 * 1000 * 1000


def load_tf_records(filenames, no_threads):
	if type(filenames) is str:
		filenames = [filenames]
	return tf.data.TFRecordDataset(filenames, buffer_size=32768, num_parallel_reads=no_threads)


def parse_example(serialized_example):
	data_fields = {
		"q_id": tf.io.VarLenFeature(tf.int64),
		"feature": tf.io.VarLenFeature(tf.float32),
		"labels": tf.io.VarLenFeature(tf.int64)

	}
	parsed = tf.io.parse_single_example(serialized_example, data_fields)
	q_id = tf.cast(tf.sparse.to_dense(parsed["q_id"]), tf.int64)[0]
	feature = tf.cast(tf.sparse.to_dense(parsed["feature"]), tf.float32)
	labels = tf.cast(tf.sparse.to_dense(parsed["labels"]), tf.float32)

	return q_id, feature, labels


def make_dataset(tf_files, no_threads):
	filenames = tf.data.Dataset.list_files(tf_files + "/*tfrecord")
	dataset = load_tf_records(filenames, no_threads)
	dataset = dataset.apply(tf.data.experimental.ignore_errors())
	dataset = dataset.map(parse_example, num_parallel_calls=no_threads)
	return dataset


def pairwise_batch_iterator(tf_records,
							window_size=1024,
							batch_size=64,
							no_threads=12,
							num_epochs=50):
	dataset = make_dataset(tf_records, no_threads)

	dataset = dataset.apply(tf.data.experimental.group_by_window(
		key_func=lambda elem, *args: elem,
		reduce_func=lambda _, window: window.batch(batch_size),
		window_size=window_size))

	# dataset = dataset.filter(lambda x, *args: tf.shape(x)[0] >= 2)
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return dataset
