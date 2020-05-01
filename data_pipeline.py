import collections
import tensorflow as tf

_READ_RECORD_BUFFER = 8 * 1000 * 1000


def load_tf_records(filenames, no_threads):
    if type(filenames) is str:
        filenames = [filenames]
    return tf.data.TFRecordDataset(filenames, buffer_size=32768, num_parallel_reads=no_threads)


def parse_example(serialized_example, data_type=tf.int32):
    data_fields = {
        "q_id": tf.io.VarLenFeature(tf.int64),
        "q": tf.io.VarLenFeature(tf.float32),
        "s1": tf.io.VarLenFeature(tf.float32),
        "add": tf.io.VarLenFeature(tf.float32),
        "label": tf.io.VarLenFeature(tf.float32)

    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    q_id = tf.cast(tf.sparse.to_dense(parsed["q_id"]), tf.int64)[0]
    q = tf.cast(tf.sparse.to_dense(parsed["q"]), tf.float32)
    doc = tf.cast(tf.sparse.to_dense(parsed["s1"]), tf.float32)

    add = tf.cast(tf.sparse.to_dense(parsed["s1_add"]), tf.float32)
    target = tf.cast(tf.sparse.to_dense(parsed["label_p"]), tf.float32)

    return q_id, q, doc, add, target


def make_dataset(tf_files, no_threads):
    dataset = load_tf_records(tf_files, no_threads)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(parse_example, num_parallel_calls=no_threads)
    return dataset


def batch_iterator(tf_records,
                   no_threads=12,
                   batch_size=16,
                   num_epochs=100,
                   shuffle=False,
                   shuffle_buffer_size=50000):
    dataset = make_dataset(tf_records, no_threads)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def file_as_batch(filename):
    dataset = tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER,
                                      num_parallel_reads=12)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(parse_example, num_parallel_calls=12)
    dataset = dataset.shuffle(buffer_size=786)
    return dataset.batch(batch_size=786)


def pairwise_batch_iterator(tf_records_path,
                            parallelism=14,
                            min_batch=64,
                            num_epochs=100):
    filenames = tf.data.Dataset.list_files(tf_records_path + "/*tfrecord")
    indices = tf.data.Dataset.range(parallelism)

    def make_dataset_lc(shard_index):
        data = filenames.shard(parallelism, shard_index)
        data = data.flat_map(file_as_batch)
        data = data.filter(lambda x, *args: tf.shape(x)[0] >= min_batch)
        return data.repeat(num_epochs)

    dataset = indices.interleave(make_dataset_lc, num_parallel_calls=parallelism)
    dataset = dataset.prefetch(buffer_size=128)

    return dataset


def pairwise_batch_iterator_2(tf_records,
                              no_threads=12,
                              batch_size=64,
                              num_epochs=50):
    dataset = make_dataset(tf_records, no_threads)

    dataset = dataset.apply(tf.data.experimental.group_by_window(
        key_func=lambda elem, *args: elem,
        reduce_func=lambda _, window: window.batch(batch_size),
        window_size=batch_size))

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
