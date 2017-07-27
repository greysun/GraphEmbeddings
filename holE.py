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


class HolEData(object):
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


def init_data():
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
        print '\t', k, np.random.choice(v, 10)

    with tf.name_scope('input'):
        # Load triples from triple_file TSV
        reader = tf.TextLineReader()
        # TODO: shard files, use TfrecordReader
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


def init_embedding(name, entity_count):
    embedding = tf.get_variable(name, [entity_count, FLAGS.embedding_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    return embedding


def get_embedding(layer_name, entity_ids, embeddings):
    entity_embeddings = tf.reshape(tf.nn.embedding_lookup(embeddings, entity_ids, max_norm=1),
                                   [-1, FLAGS.embedding_dim])
    # TODO: subtract the mean for entities of a given type
    embeddings = tf.slice(entity_embeddings, [0, 0], [-1, FLAGS.embedding_dim])
    return tf.reshape(embeddings, [-1, FLAGS.embedding_dim], name=layer_name)


def reduce_tanh(batch_tensor):
    return tf.tanh(tf.reduce_sum(batch_tensor, 1, keep_dims=True))


def circular_correlation(h, t):
    # these ops are GPU only!
    return tf.spectral.irfft(tf.multiply(tf.conj(tf.spectral.rfft(h)), tf.spectral.rfft(t)))


def evaluate_triples(triple_batch, embeddings, label=None):
    # Load embeddings
    with tf.device('/cpu:0'):
        head_column = tf.slice(triple_batch, [0, 0], [-1, 1], name='h_id')
        head_embeddings = get_embedding('h', head_column, embeddings)
        tail_column = tf.slice(triple_batch, [0, 1], [-1, 1], name='t_id')
        tail_embeddings = get_embedding('t', tail_column, embeddings)
        relation_column = tf.slice(triple_batch, [0, 2], [-1, 1], name='r_id')
        relation_embeddings = get_embedding('r', relation_column, embeddings)

    # Compute loss
    with tf.name_scope('eval'):
        if FLAGS.cpu:
            # TransE
            score = head_embeddings + relation_embeddings - tail_embeddings
        else:
            score = tf.multiply(relation_embeddings, circular_correlation(head_embeddings, tail_embeddings))

        if FLAGS.log_loss and label:
            score = tf.scalar_mul(-label, score)
            loss = tf.log(1. + tf.exp(score))
            # TODO: regularization
        else:
            loss = reduce_tanh(score)

        summarize(loss)

    return loss


def summarize(var):
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
    batch_count = data.triple_count / FLAGS.batch_size
    print 'Embedding dimension: ', FLAGS.embedding_dim, 'Batch size: ', FLAGS.batch_size, 'Batch count: ', batch_count

    # Warning: this will clobber existing summaries
    if not FLAGS.resume_checkpoint and os.path.isdir(FLAGS.output_dir):
        shutil.rmtree(FLAGS.output_dir)
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Initialize embeddings
    embeddings = init_embedding('embeddings', data.entity_count)

    # Initialize tables for type-safe corruption (to avoid junk triples like 'Jeff', 'Employer', 'Java')
    with tf.name_scope('corruption_tables'):
        with tf.name_scope('type_to_ids'):
            type_to_ids_table = init_table(tf.string, tf.int64, 'type_to_ids', type_to_ids=True)
            type_to_ids_keys = tf.placeholder(tf.string, [len(data.type_to_ids)], 'keys')
            type_to_ids_values = tf.placeholder(tf.int64, [len(data.type_to_ids), FLAGS.padded_size], 'values')
            type_to_ids_insert = type_to_ids_table.insert(type_to_ids_keys, type_to_ids_values)
        with tf.name_scope('id_to_type'):
            id_to_type_table = init_table(tf.int64, tf.string, 'id_to_type')
            id_to_type_keys = tf.placeholder(tf.int64, [data.entity_count], 'keys')
            id_to_type_values = tf.placeholder(tf.string, [data.entity_count], 'values')
            id_to_type_insert = id_to_type_table.insert(id_to_type_keys, id_to_type_values)

    with tf.name_scope('batch'):
        # Sample triples
        triple_batch = tf.train.shuffle_batch([data.triples], FLAGS.batch_size,
                                              num_threads=FLAGS.reader_threads,
                                              capacity=2*data.triple_count,
                                              # TODO: this probably won't scale
                                              min_after_dequeue=data.triple_count,
                                              allow_smaller_final_batch=False,
                                              name="shuffle_batch")

        # Evaluate triples
        if FLAGS.log_loss:
            losses = []
            with tf.name_scope('train'):
                train_loss = evaluate_triples(triple_batch, embeddings, 1)
                # Increase the weight of the positive case
                # TODO: decrease this weight over time
                losses.append(tf.scalar_mul(FLAGS.negative_ratio, train_loss))
            with (tf.name_scope('corrupt')):
                for i in range(FLAGS.negative_ratio):
                    with tf.name_scope('c' + str(i)):
                        corrupt_triples = corrupt_batch(type_to_ids_table, id_to_type_table,
                                                        data.relation_count, triple_batch)
                        corrupt_loss = evaluate_triples(corrupt_triples, embeddings, -1)
                        losses.append(corrupt_loss)

            # Minimize log-loss
            loss = tf.concat(losses, 0)
        else:
            with tf.name_scope('train'):
                train_loss = evaluate_triples(triple_batch, embeddings)
            with (tf.name_scope('corrupt')):
                corrupt_triples = corrupt_batch(type_to_ids_table, id_to_type_table, data.relation_count, triple_batch)
                corrupt_loss = evaluate_triples(corrupt_triples, embeddings)

            # Score and minimize hinge-loss
            loss = tf.maximum(train_loss - corrupt_loss + FLAGS.margin, 0)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr_decay = tf.train.inverse_time_decay(FLAGS.learning_rate, global_step,
                                               decay_steps=FLAGS.learning_decay_steps*batch_count,
                                               decay_rate=FLAGS.learning_decay_rate)
        tf.summary.scalar('learning_rate', lr_decay)
        optimizer = tf.train.GradientDescentOptimizer(lr_decay).minimize(loss, global_step)

        tf.summary.histogram('embeddings', embeddings)

    summaries = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
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
                projector.visualize_embeddings(summary_writer, projector_config)

                # Shuffle the available corrupt entity ids and insert every epoch
                padded_values = np.array([[random.choice(v) for _ in range(FLAGS.padded_size)]
                                          for v in data.type_to_ids.values()])
                feed_dict = {type_to_ids_keys: np.array(data.type_to_ids.keys()),
                             type_to_ids_values: np.array(padded_values)}
                sess.run([type_to_ids_insert], feed_dict)

                for batch in range(1, batch_count):
                    # TODO: print eval scores from validation
                    # Train and log batch to summary_writer
                    if batch % (batch_count / 16) == 0:
                        model, summary = sess.run([optimizer, summaries])
                        step = (epoch - 1) * batch_count + batch
                        summary_writer.add_summary(summary, step)
                        print '\tSaved summary for step {}...'.format(step)

                    else:
                        sess.run([optimizer])

                # Checkpoint
                # TODO: only checkpoint if validation MRR decreases
                save_path = saver.save(sess, FLAGS.output_dir + '/model.ckpt', epoch)
                print('Epoch {}, (Model saved as {})'.format(epoch, save_path))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            saver.save(sess, FLAGS.output_dir + '/model.ckpt')

        coord.join(threads)


class HolEInferenceData(HolEData):
    def __init__(self):
        self.id_to_metadata = dict()
        # TODO: this won't scale up to large datasets
        self.true_triples = defaultdict(lambda: defaultdict(set))
        self.test_triples = defaultdict(lambda: defaultdict(set))
        super(HolEInferenceData, self).__init__()


def init_inference_data():
    entity_file = os.path.join(FLAGS.data_dir, 'entity_metadata.tsv')
    relation_file = os.path.join(FLAGS.data_dir, 'relation_ids.txt')
    train_triples = os.path.join(FLAGS.data_dir, 'triples.txt')
    valid_triples = os.path.join(FLAGS.data_dir, 'triples-valid.txt')
    test_triples = os.path.join(FLAGS.data_dir, 'test_positive_triples.txt')

    data = HolEInferenceData()

    with open(entity_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            data.entity_count += 1
            index, entity_id, name, entity_type = line.strip().split('\t')
            index = int(index)
            data.type_to_ids[entity_type].append(index)
            data.id_to_metadata[index] = entity_id + ' ' + name

    data.relation_count = sum(1 for line in open(relation_file))

    print 'Types: ', {k: len(v) for k, v in data.type_to_ids.iteritems()}

    with open(test_triples, 'r') as f:
        for line in f:
            head_id, tail_id, relation_id = line.strip().split('\t')
            head_id = int(head_id)
            tail_id = int(tail_id)
            relation_id = int(relation_id)
            data.test_triples[head_id][relation_id].add(tail_id)

    for triple_file in [train_triples, valid_triples]:
        with open(triple_file, 'r') as f:
            for line in f:
                head_id, tail_id, relation_id = line.strip().split('\t')
                head_id = int(head_id)
                tail_id = int(tail_id)
                relation_id = int(relation_id)

                if relation_id in data.test_triples[head_id]:
                    data.true_triples[head_id][relation_id].add(tail_id)

    return data


def eval_link_prediction(scores, id_to_metadata, true_triples, test_triples, raw_positions, filtered_positions):
    heap = []
    for pair in scores:
        loss = pair[0][0]
        heappush(heap, (loss, tuple(pair[1])))
        person_id = id_to_metadata[pair[1][0]]
        relation = id_to_metadata[pair[1][2]]

    # TODO: adapt format for Diffbot/FB15k
    print 'https://diffbot.com/entity/' + person_id, relation

    raw_rank = 0
    filtered_rank = 0
    while heap:
        pair = heappop(heap)
        loss = pair[0]
        head_id = pair[1][0]
        tail_id = pair[1][1]
        relation_id = pair[1][2]

        raw_rank += 1
        if tail_id in true_triples[head_id][relation_id]:
            print '\tTRAIN {}: {}\thttps://diffbot.com/entity/{}'.format(raw_rank, loss, id_to_metadata[tail_id])
            continue
        elif tail_id in test_triples[head_id][relation_id]:
            raw_positions.append(raw_rank)

        filtered_rank += 1
        if tail_id in test_triples[head_id][relation_id]:
            filtered_positions.append(filtered_rank)
            print '\tMATCH {}: {}\thttps://diffbot.com/entity/{}'.format(filtered_rank, loss, id_to_metadata[tail_id])
            continue
        elif filtered_rank < 5:
            print '\tGUESS {}: {}\thttps://diffbot.com/entity/{}'.format(filtered_rank, loss, id_to_metadata[tail_id])


def infer_triples():
    data = init_inference_data()
    # Consider locations current location, location (region, country)
    #relation_ids = [10, 13, 14]
    relation_ids = [9, 10]
    tail_type = 'A'

    with tf.name_scope('inference'):
        embeddings = init_embedding('embeddings', data.entity_count)

        triple_batch = tf.placeholder(tf.int64, [len(data.type_to_ids[tail_type]), 3], 'triples') \
            if FLAGS.infer_typesafe else tf.placeholder(tf.int64, [data.entity_count-data.relation_count, 3], 'triples')
        eval_loss = evaluate_triples(triple_batch, embeddings)

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
                raw_positions = []
                filtered_positions = []

                for head in data.test_triples:
                    for relation in data.test_triples[head]:
                        if FLAGS.infer_typesafe and relation not in relation_ids:
                            continue

                        candidate_tails = data.type_to_ids[tail_type] if FLAGS.infer_typesafe else \
                            range(data.relation_count, data.entity_count)
                        candidate_triples = np.array(list(itertools.product([head], candidate_tails, [relation])))
                        feed_dict = {triple_batch: candidate_triples}
                        triples, batch_loss = sess.run([triple_batch, eval_loss], feed_dict)

                        eval_link_prediction(zip(batch_loss, triples), data.id_to_metadata,
                                             data.true_triples, data.test_triples,
                                             raw_positions, filtered_positions)

                raw_positions = np.array(raw_positions)
                raw_mrr = np.mean(1.0 / raw_positions)
                mean_raw_pos = np.mean(raw_positions)

                filtered_positions = np.array(filtered_positions)
                filtered_mrr = np.mean(1.0 / filtered_positions)
                mean_filtered_pos = np.mean(filtered_positions)

                hits1 = np.mean(filtered_positions <= 1).sum() * 100
                hits3 = np.mean(filtered_positions <= 3).sum() * 100
                hits10 = np.mean(filtered_positions <= 10).sum() * 100
                print '\n\n\nRaw MRR: {} (mean position: {})'.format(raw_mrr, mean_raw_pos)
                print 'Filtered MRR: {} (mean position: {})'.format(filtered_mrr, mean_filtered_pos)
                print 'Hits at 1: {}, 3: {}, 10: {}'.format(hits1, hits3, hits10)

            except tf.errors.OutOfRangeError:
                print('Done evaluation -- triple limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


def main(_):
    # TODO: refactor model in to object
    if FLAGS.infer or FLAGS.infer_typesafe:
        infer_triples()
    else:
        training_data = init_data()
        # TODO: param search
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
        default=0.1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--learning_decay_steps',
        type=float,
        default=16,
        help='Learning rate decay steps (in epochs).'
    )
    parser.add_argument(
        '--learning_decay_rate',
        type=float,
        default=0.5,
        help='Learning decay rate.'
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
        help='Number of negative labels to sample for each positive label (log_loss only).'
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
        '--infer_typesafe',
        action='store_true',
        help='Infer new triples using type-safe candidate entities only.'
    )
    parser.add_argument(
        '--inference-threshold',
        type=float,
        default=0.8,
        help='Infer new triples from the latest checkpoint model.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

