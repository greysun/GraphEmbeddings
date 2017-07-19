"""
Tensorflow implementation of Nickel et al, HolE, 2016

See https://arxiv.org/pdf/1510.04935.pdf
"""
import argparse
import errno
import itertools
import os
import random
import shutil
import sys
from heapq import heappush, heappop

import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug


FLAGS = None


class HolEData:
    def __init__(self):
        self.type_to_ids = defaultdict(list)
        self.id_to_type = dict()
        self.entity_count = 0
        self.relation_count = 0
        self.triple_count = 0
        self.triples = None


def init_table(key_dtype, value_dtype, name, type_to_ids=False):
    """Initializes the table variable and all of the inputs as constants."""
    default_value = tf.constant(FLAGS.padded_size * [-1], value_dtype) if type_to_ids else tf.constant('?', value_dtype)
    return tf.contrib.lookup.MutableHashTable(key_dtype=key_dtype, value_dtype=value_dtype,
                                              default_value=default_value, shared_name=name, name=name)


def get_the_data():
    entity_file = os.path.join(FLAGS.data_dir, 'entity_metadata.tsv')
    relation_file = os.path.join(FLAGS.data_dir, 'relation_ids.txt')
    corrupt_triple_file = os.path.join(FLAGS.data_dir, 'triples.txt')

    data = HolEData()

    data.relation_count = sum(1 for line in open(relation_file))
    data.triple_count = sum(1 for line in open(corrupt_triple_file))

    with open(entity_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            data.entity_count += 1
            index, diffbot_id, name, diffbot_type = line.split('\t')
            index = int(index)
            diffbot_type = diffbot_type.strip()
            data.type_to_ids[diffbot_type].append(index)
            data.id_to_type[index] = diffbot_type

    print 'Entities: ', data.entity_count - data.relation_count, 'Relations: ', data.relation_count, \
        'Triples: ', data.triple_count
    print 'Types: ', {k: len(v) for k, v in data.type_to_ids.iteritems()}
    for k, v in data.type_to_ids.iteritems():
        print "\t", k, np.random.choice(v, 10)

    with tf.name_scope('input'):
        # Load triples from triple_file TSV
        reader = tf.TextLineReader()
        # TODO: shard files, TfrecordReader
        # TODO: tf.records, move preprocessing to separate script
        filename_queue = tf.train.string_input_producer([corrupt_triple_file] * FLAGS.num_epochs)
        key, value = reader.read(filename_queue)

        column_defaults = [tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32)]

        head_ids, tail_ids, relation_ids = tf.decode_csv(value, column_defaults, field_delim='\t')
        data.triples = tf.stack([head_ids, tail_ids, relation_ids])

    return data


def corrupt_heads(type_to_ids, id_to_type, triples):
    # TODO: need to avoid same type entities for relation 'instance_of'
    with tf.name_scope('head'):
        head_column = tf.cast(tf.slice(triples, [0, 0], [-1, 1]), tf.int64)
        tail_column = tf.slice(triples, [0, 1], [-1, 1])
        relation_column = tf.slice(triples, [0, 2], [-1, 1])

        head_types = id_to_type.lookup(head_column)
        type_ids = tf.reshape(type_to_ids.lookup(head_types), [-1, FLAGS.padded_size])

        size = tf.shape(head_column)[0]
        random_indices = tf.random_uniform([size],
                                           maxval=FLAGS.padded_size,
                                           dtype=tf.int32)
        flattened_indices = tf.range(0, size) * FLAGS.padded_size + random_indices
        corrupt_head_column = tf.reshape(tf.gather(tf.reshape(type_ids, [-1]), flattened_indices), [size, 1])
        concat = tf.concat([tf.cast(corrupt_head_column, tf.int32), tail_column, relation_column], 1)
        return concat


def corrupt_tails(type_to_ids, id_to_type, triples):
    with tf.name_scope('tail'):
        head_column = tf.slice(triples, [0, 0], [-1, 1])
        tail_column = tf.cast(tf.slice(triples, [0, 1], [-1, 1]), tf.int64)
        relation_column = tf.slice(triples, [0, 2], [-1, 1])

        tail_types = id_to_type.lookup(tail_column)
        type_ids = tf.reshape(type_to_ids.lookup(tail_types), [-1, FLAGS.padded_size])

        size = tf.shape(head_column)[0]
        random_indices = tf.random_uniform([size],
                                           maxval=FLAGS.padded_size,
                                           dtype=tf.int32)
        flattened_indices = tf.range(0, size) * FLAGS.padded_size + random_indices
        corrupt_tail_column = tf.reshape(tf.gather(tf.reshape(type_ids, [-1]), flattened_indices), [size, 1])
        concat = tf.concat([head_column, tf.cast(corrupt_tail_column, tf.int32), relation_column], 1)
        return concat


def corrupt_entities(type_to_ids, id_to_type, triples):
    should_corrupt_heads = tf.less(tf.random_uniform([], 0, 1.0), 0.5, 'should_corrupt_heads')
    return tf.cond(should_corrupt_heads,
                   lambda: corrupt_heads(type_to_ids, id_to_type, triples),
                   lambda: corrupt_tails(type_to_ids, id_to_type, triples))


def corrupt_relations(relation_count, triples):
    with tf.name_scope('relation'):
        entity_columns = tf.slice(triples, [0, 0], [-1, 2])
        corrupt_relation_column = tf.random_uniform([tf.shape(entity_columns)[0], 1],
                                                    maxval=relation_count,
                                                    dtype=tf.int32)
        return tf.concat([entity_columns, corrupt_relation_column], 1)


def corrupt_batch(type_to_ids, id_to_type, relation_count, triples):
    should_corrupt_relations = tf.less(tf.random_uniform([], 0, 1.0), 0.5, 'should_corrupt_relations')
    return tf.cond(should_corrupt_relations,
                   lambda: corrupt_relations(relation_count, triples),
                   lambda: corrupt_entities(type_to_ids, id_to_type, triples))


def init_embedding(name, entity_count, embedding_dim):
    embedding = tf.get_variable(name, [entity_count, 2*embedding_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    return embedding


def get_embedding(layer_name, entity_ids, embeddings, embedding_dim):
    entity_embeddings = tf.reshape(tf.nn.embedding_lookup(embeddings, entity_ids, max_norm=1),
                                   [-1, 2*embedding_dim])
    real_embeddings = tf.slice(entity_embeddings, [0, 0], [-1, embedding_dim])
    imag_embeddings = tf.slice(entity_embeddings, [0, embedding_dim], [-1, embedding_dim]),
    return tf.reshape(tf.complex(real_embeddings, imag_embeddings), [-1, embedding_dim], name=layer_name)


def complex_tanh(complex_tensor):
    summed = tf.reduce_sum(tf.real(complex_tensor) + tf.imag(complex_tensor), 1, keep_dims=True)
    return tf.tanh(summed)


def circular_correlation(h, t):
    # these ops are GPU only!
    return tf.ifft(tf.multiply(tf.conj(tf.fft(h)), tf.fft(t)))


def evaluate_triples(triple_batch, embeddings, embedding_dim, label=None):
    # Load embeddings
    with tf.device('/cpu:0'):
        head_column = tf.slice(triple_batch, [0, 0], [-1, 1], name='h_id')
        head_embeddings = get_embedding('h', head_column, embeddings, embedding_dim)
        tail_column = tf.slice(triple_batch, [0, 1], [-1, 1], name='t_id')
        tail_embeddings = get_embedding('t', tail_column, embeddings, embedding_dim)
        relation_column = tf.slice(triple_batch, [0, 2], [-1, 1], name='r_id')
        relation_embeddings = get_embedding('r', relation_column, embeddings, embedding_dim)

    # Compute loss
    with tf.name_scope('eval'):
        if FLAGS.cpu:
            # TransE
            score = head_embeddings + relation_embeddings - tail_embeddings
        else:
            score = tf.multiply(relation_embeddings, circular_correlation(head_embeddings, tail_embeddings))

        if FLAGS.log_loss and label:
            complex_score = tf.scalar_mul(-label, score)
            real_score = tf.real(complex_score) + tf.imag(complex_score)
            loss = tf.log(1. + tf.exp(real_score))
            # TODO: regularization
        else:
            loss = complex_tanh(score)

        variable_summaries(loss)

    return loss


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def run_training(data):
    # Initialize parameters
    margin = FLAGS.margin
    embedding_dim = FLAGS.embedding_dim
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    batch_count = data.triple_count / batch_size
    print 'Embedding dimension: ', embedding_dim, 'Batch size: ', batch_size, 'Batch count: ', batch_count

    # Warning: this will clobber existing summaries
    if not FLAGS.resume_checkpoint and os.path.isdir(FLAGS.output_dir):
        shutil.rmtree(FLAGS.output_dir)
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with tf.device('/cpu'):
        # Initialize embeddings (TF doesn't support complex embeddings, split real part and imaginary part)
        embeddings = init_embedding('embeddings', data.entity_count, embedding_dim)

        # Initialize tables for type-safe corruption (to avoid junk triples like "Jeff", "Employer", "Java")
        with tf.name_scope('tables'):
            with tf.name_scope('type_to_ids'):
                type_to_ids_keys = tf.placeholder(tf.string, [len(data.type_to_ids)], 'keys')
                type_to_ids_values = tf.placeholder(tf.int64, [len(data.type_to_ids), FLAGS.padded_size], 'values')

                type_to_ids_table = init_table(tf.string, tf.int64, 'type_to_ids', type_to_ids=True)
                type_to_ids_insert = type_to_ids_table.insert(type_to_ids_keys, type_to_ids_values)
            with tf.name_scope('id_to_type'):
                id_to_type_keys = tf.placeholder(tf.int64, [data.entity_count], 'keys')
                id_to_type_values = tf.placeholder(tf.string, [data.entity_count], 'values')

                id_to_type_table = init_table(tf.int64, tf.string, 'id_to_type')
                id_to_type_insert = id_to_type_table.insert(id_to_type_keys, id_to_type_values)

    with tf.name_scope('batch'):
        # Sample triples
        triple_batch = tf.train.shuffle_batch([data.triples], batch_size,
                                              num_threads=FLAGS.reader_threads,
                                              capacity=2*data.triple_count,
                                              # TODO: this probably won't scale
                                              min_after_dequeue=data.triple_count,
                                              allow_smaller_final_batch=False)

        # Evaluate triples
        if FLAGS.log_loss:
            losses = []
            with tf.name_scope('train'):
                train_loss = evaluate_triples(triple_batch, embeddings, embedding_dim, 1)
                # Increase the weight of the positive case
                # TODO: decrease this weight over time
                losses.append(tf.scalar_mul(FLAGS.negative_ratio, train_loss))
            with (tf.name_scope('corrupt')):
                for i in range(FLAGS.negative_ratio):
                    with tf.name_scope('c' + str(i)):
                        corrupt_triples = corrupt_batch(type_to_ids_table, id_to_type_table,
                                                        data.relation_count, triple_batch)
                        corrupt_loss = evaluate_triples(corrupt_triples, embeddings, embedding_dim, -1)
                        losses.append(corrupt_loss)

            # Minimize log-loss
            loss = tf.concat(losses, 0)
        else:
            with tf.name_scope('train'):
                train_loss = evaluate_triples(triple_batch, embeddings, embedding_dim)
            with (tf.name_scope('corrupt')):
                corrupt_triples = corrupt_batch(type_to_ids_table, id_to_type_table, data.relation_count, triple_batch)
                corrupt_loss = evaluate_triples(corrupt_triples, embeddings, embedding_dim)

            # Score and minimize hinge-loss
            loss = tf.maximum(train_loss - corrupt_loss + margin, 0)

        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    summaries = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()

    # Save embeddings
    saver = tf.train.Saver({'embeddings': embeddings})

    supervisor = tf.train.Supervisor(logdir=FLAGS.output_dir)
    with supervisor.managed_session() as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        sess.run(init_op)

        # Load the previous model if resume_checkpoint=True
        if FLAGS.resume_checkpoint:
            saver.restore(sess, FLAGS.output_dir + '/model.ckpt')
            # TODO: continue counting from last epoch

        summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

        # TODO: separate thread for shuffling type_to_ids
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            # Populate id_to_type mapping
            feed_dict = {id_to_type_keys: np.array(data.id_to_type.keys()),
                         id_to_type_values: np.array(data.id_to_type.values())}
            sess.run([id_to_type_insert], feed_dict)

            epoch = 0
            while not supervisor.should_stop():
                epoch += 1

                projector_config = projector.ProjectorConfig()
                embeddings_config = projector_config.embeddings.add()
                embeddings_config.tensor_name = embeddings.name
                embeddings.metadata_path = os.path.join(FLAGS.data_dir, 'entity_metadata.tsv')
                projector.visualize_embeddings(summary_writer, projector_config)

                # Shuffle the available corrupt entity ids and insert every epoch
                padded_values = np.array([[random.choice(v) for _ in range(FLAGS.padded_size)]
                                          for v in data.type_to_ids.values()])
                feed_dict = {type_to_ids_keys: np.array(data.type_to_ids.keys()),
                             type_to_ids_values: np.array(padded_values)}
                sess.run([type_to_ids_insert], feed_dict)

                for batch in range(1, batch_count):
                    # Train and log batch to summary_writer
                    if batch % (batch_count / 16) == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        model, summary = sess.run(
                            [optimizer, summaries],
                            options=run_options,
                            run_metadata=run_metadata)
                        step = '{}-{}'.format(epoch, batch)
                        summary_writer.add_run_metadata(run_metadata, step)
                        summary_writer.add_summary(summary, epoch * batch_count + batch)
                        print '\tSaved summary for step {}...'.format(step)

                    else:
                        sess.run([optimizer])

                # Checkpoint
                save_path = saver.save(sess, FLAGS.output_dir + '/model.ckpt', epoch)
                print('Epoch {}, (Model saved as {})'.format(epoch, save_path))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            saver.save(sess, FLAGS.output_dir + '/model.ckpt')

        coord.join(threads)


def infer_triples():
    # TODO: this should be loaded from the saved model
    embedding_dim = FLAGS.embedding_dim
    entity_file = os.path.join(FLAGS.data_dir, 'entity_metadata.tsv')
    skill_train = os.path.join(FLAGS.data_dir, 'train_skills.txt')
    skill_test = os.path.join(FLAGS.data_dir, 'test_skills.txt')

    type_to_ids = defaultdict(list)
    id_to_metadata = dict()

    entity_count = 0
    with open(entity_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            entity_count += 1
            index, diffbot_id, name, diffbot_type = line.strip().split('\t')
            index = int(index)
            type_to_ids[diffbot_type].append(index)
            id_to_metadata[index] = diffbot_id + ' ' + name

    print 'Types: ', {k: len(v) for k, v in type_to_ids.iteritems()}

    infer_heads = set()
    train_skills = defaultdict(list)
    test_skills = defaultdict(list)

    with open(skill_test, 'r') as f:
        for line in f:
            head_id, skill_id, _ = line.strip().split('\t')
            infer_heads.add(head_id)
            test_skills[head_id].append(skill_id)

    with open(skill_train, 'r') as f:
        for line in f:
            head_id, skill_id, _ = line.strip().split('\t')
            if head_id in test_skills:
                train_skills[head_id].append(skill_id)

    # Infer skills
    infer_relations = [6]

    # Candidate targets
    infer_tails = type_to_ids['S']

    with tf.name_scope('inference'):
        embeddings = init_embedding('embeddings', entity_count, embedding_dim)

        triple_batch = tf.placeholder(tf.int64, [len(infer_tails), 3], 'triples')
        eval_loss = evaluate_triples(triple_batch, embeddings, embedding_dim)

        # Load embeddings
        saver = tf.train.Saver({'embeddings': embeddings})

        with tf.Session() as sess:
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            saver.restore(sess, FLAGS.output_dir + '/model.ckpt')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                for head in infer_heads:
                    candidate_triples = np.array(list(itertools.product([head], infer_tails, infer_relations)))
                    feed_dict = {triple_batch: candidate_triples}
                    triples, batch_loss = sess.run([triple_batch, eval_loss], feed_dict)

                    heap = []
                    for pair in zip(batch_loss, triples):
                        loss = pair[0][0]
                        heappush(heap, (loss, tuple(pair[1])))
                        person_id = id_to_metadata[pair[1][0]]

                    print 'https://diffbot.com/entity/' + person_id + ' skills:'
                    for i in range(10):
                        # TODO: score hits at 1,3,10
                        # TODO: score raw/filtered MRR
                        pair = heappop(heap)
                        loss = pair[0]
                        head_id = pair[1][0]
                        skill_id = pair[1][1]

                        match = ""
                        if skill_id in test_skills[head_id]:
                            match = "MATCH"
                        elif skill_id in train_skills[head_id]:
                            match = "TRAIN"

                        print '\t{}\thttps://diffbot.com/entity/{} {}'.format(loss, id_to_metadata[skill_id], match)

            except tf.errors.OutOfRangeError:
                print('Done evaluation -- triple limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


def main(_):
    # TODO: refactor model in to object
    if FLAGS.infer:
        infer_triples()
    else:
        training_data = get_the_data()
        run_training(training_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Disable GPU-only operations (namely FFT/iFFT).'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=128,
        help='Embedding dimension.'
    )
    parser.add_argument(
        '--log_loss',
        action='store_true',
        help='Use logistic loss. Otherwise, use pairwise ranking loss.'
    )
    parser.add_argument(
        '--negative_ratio',
        type=int,
        default=4,
        help='Number of negative labels to sample for each positive label.'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.,
        help='Hinge loss margin.'
    )
    parser.add_argument(
        '--padded_size',
        type=int,
        default=100000,
        help='The maximum number of entities to use for each type while sampling corrupt triples.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='holE-latest',
        help='Tensorboard Summary directory.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='diffbot_data/kg_0.01',
        help='Input data directory. Must contain {triples.txt, entity_metadata.tsv, relation_ids.txt}'
    )
    parser.add_argument(
        '--reader_threads',
        type=int,
        default=4,
        help='Number of training triple file readers.'
    )
    parser.add_argument(
        '--resume_checkpoint',
        action='store_true',
        help='Resume training on the checkpoint model. Otherwise start with randomly initialized embeddings.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run with interactive Tensorflow debugger.'
    )
    parser.add_argument(
        '--infer',
        action='store_true',
        help='Infer new triples from the latest checkpoint model.'
    )
    parser.add_argument(
        '--inference-threshold',
        type=float,
        default=0.8,
        help='Infer new triples from the latest checkpoint model.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

