"""
Tensorflow implementation of Nickel et al, HolE, 2016

See https://arxiv.org/pdf/1510.04935.pdf

Author: B Han
"""
import argparse
import errno
import itertools
import os
import random
import sys
from heapq import heappush, heappop

import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.tensorboard.plugins import projector


FLAGS = None


class HolEData(object):
    """Pre-processing data used during training and inference."""
    def __init__(self):
        self.type_to_ids = defaultdict(list)
        self.id_to_type = dict()
        self.entity_count = 0
        self.relation_count = 0
        self.triple_count = 0
        self.triples = None
        self.validation_triples = None


def init_table(key_dtype, value_dtype, name, type_to_ids=False):
    """Initializes a TF table variable with the appropriate default_value."""
    default_value = tf.constant(FLAGS.padded_size * [-1], value_dtype) if type_to_ids else tf.constant('?', value_dtype)
    return tf.contrib.lookup.MutableHashTable(key_dtype=key_dtype, value_dtype=value_dtype,
                                              default_value=default_value, shared_name=name, name=name)


def init_data():
    """Model pre-processing."""
    entity_file = os.path.join(FLAGS.data_dir, 'entity_metadata.tsv')
    relation_file = os.path.join(FLAGS.data_dir, 'relation_ids.txt')
    train_triple_file = os.path.join(FLAGS.data_dir, 'triples.txt')
    valid_triple_file = os.path.join(FLAGS.data_dir, 'triples-valid.txt')

    data = HolEData()
    data.relation_count = sum(1 for line in open(relation_file))
    data.triple_count = sum(1 for line in open(train_triple_file))

    with open(entity_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            data.entity_count += 1
            index, diffbot_id, name, diffbot_type, mentions, is_tail = line.strip().split('\t')
            index = int(index)
            data.type_to_ids[diffbot_type].append(index)
            data.id_to_type[index] = diffbot_type

    print('Entities: ', data.entity_count - data.relation_count, 'Relations: ', data.relation_count,
          'Triples: ', data.triple_count)
    print('Types: ', {k: len(v) for k, v in data.type_to_ids.items()})
    for k, v in data.type_to_ids.items():
        print('\t', k, np.random.choice(v, 10))

    with tf.name_scope('input'):
        # Load triples from triple_file TSV
        reader = tf.TextLineReader()
        # TODO: shard files, use TfrecordReader
        filename_queue = tf.train.string_input_producer([train_triple_file] * FLAGS.num_epochs)
        key, value = reader.read(filename_queue)
        column_defaults = [tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32)]

        head_ids, tail_ids, relation_ids = tf.decode_csv(value, column_defaults, field_delim='\t')
        data.triples = tf.stack([head_ids, tail_ids, relation_ids])

    with tf.name_scope('valid_input'):
        reader = tf.TextLineReader()
        filename_queue = tf.train.string_input_producer([valid_triple_file])
        key, value = reader.read(filename_queue)
        column_defaults = [tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32)]

        head_ids, tail_ids, relation_ids = tf.decode_csv(value, column_defaults, field_delim='\t')
        data.validation_triples = tf.stack([head_ids, tail_ids, relation_ids])

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
    return corrupt_entities(type_to_ids, id_to_type, triples)
    # TODO: consider corrupting more entities as training time increases
    #should_corrupt_relations = tf.less(tf.random_uniform([], 0, 1.0), 0.2, 'should_corrupt_relations')
    #return tf.cond(should_corrupt_relations,
    #               lambda: corrupt_relations(relation_count, triples),
    #               lambda: corrupt_entities(type_to_ids, id_to_type, triples))


def get_embedding(layer_name, entity_ids, embeddings):
    entity_embeddings = tf.reshape(tf.nn.embedding_lookup(embeddings, entity_ids, max_norm=1),
                                   [-1, FLAGS.embedding_dim])
    real_embeddings = tf.slice(entity_embeddings, [0, 0], [-1, FLAGS.embedding_dim//2])
    imag_embeddings = tf.slice(entity_embeddings, [0, FLAGS.embedding_dim//2], [-1, FLAGS.embedding_dim//2])
    complex_embeddings = tf.reshape(tf.complex(real_embeddings, imag_embeddings),
                                    [-1, FLAGS.embedding_dim//2], name=layer_name)
    return complex_embeddings


def reduce_eval(batch_tensor):
    return tf.sigmoid(tf.reduce_sum(batch_tensor, 1, keep_dims=True))


def circular_correlation(h, t):
    return tf.real(tf.multiply(tf.conj(tf.spectral.rfft(h)), tf.spectral.rfft(t)))


def evaluate_triples(triple_batch, embeddings, label=None):
    # Load embeddings
    head_column = tf.slice(triple_batch, [0, 0], [-1, 1], name='h_id')
    head_embeddings = get_embedding('h', head_column, embeddings)
    tail_column = tf.slice(triple_batch, [0, 1], [-1, 1], name='t_id')
    tail_embeddings = get_embedding('t', tail_column, embeddings)
    relation_column = tf.slice(triple_batch, [0, 2], [-1, 1], name='r_id')
    relation_embeddings = get_embedding('r', relation_column, embeddings)

    # Compute loss
    with tf.name_scope('eval'):
        # TODO: soft-regularization (instead of max_norm=1)
        score = tf.multiply(head_embeddings, tf.multiply(relation_embeddings, tf.conj(tail_embeddings)))
        score = tf.reduce_sum(tf.real(score), 1, keep_dims=True)

        if FLAGS.log_loss and label is not None:
            score = tf.scalar_mul(-label, score)
            loss = tf.log(1. + tf.exp(score)) + FLAGS.l2_regularization * tf.nn.l2_loss(embeddings)
        else:
            loss = tf.sigmoid(score)

        summarize(loss)

    return loss


def evaluate_batch(triple_batch, embeddings, type_to_ids_table, id_to_type_table, relation_count):
    if FLAGS.log_loss:
        losses = []
        with tf.name_scope('positive'):
            train_loss = evaluate_triples(triple_batch, embeddings, 1)
            losses.append(train_loss)
        with (tf.name_scope('corrupt')):
            for i in range(FLAGS.negative_ratio):
                with tf.name_scope('c' + str(i)):
                    corrupt_triples = corrupt_batch(type_to_ids_table, id_to_type_table, relation_count, triple_batch)
                    corrupt_loss = evaluate_triples(corrupt_triples, embeddings, -1)
                    losses.append(corrupt_loss)

        # TODO: support validation triples in log_loss

        return tf.concat(losses, 0)

    else:
        with tf.name_scope('positive'):
            train_loss = evaluate_triples(triple_batch, embeddings)
        with tf.name_scope('corrupt'):
            corrupt_triples = corrupt_batch(type_to_ids_table, id_to_type_table, relation_count, triple_batch)
            corrupt_loss = evaluate_triples(corrupt_triples, embeddings)

        # Score and minimize hinge-loss
        # TODO: experiment with margin growth over time
        loss = tf.maximum(train_loss - corrupt_loss + FLAGS.margin, 0, name="loss")
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
    batch_count = data.triple_count // FLAGS.batch_size
    print('Embedding dimension: ', FLAGS.embedding_dim, 'Batch size: ', FLAGS.batch_size, 'Batch count: ', batch_count)

    # Warning: this will clobber existing summaries
    if not FLAGS.resume_checkpoint and os.path.isdir(FLAGS.output_dir):
        raise Exception("WARNING: " + FLAGS.output_dir + " already exists!")
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Initialize embeddings
    embeddings = tf.get_variable('embeddings', [data.entity_count, FLAGS.embedding_dim],
                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False))

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
        triple_batch = tf.train.shuffle_batch([data.triples], FLAGS.batch_size, num_threads=FLAGS.reader_threads,
                                              capacity=2*data.triple_count, min_after_dequeue=data.triple_count,
                                              allow_smaller_final_batch=False, name='shuffle_batch')

        # Evaluate triples
        with tf.name_scope('eval'):
            loss = evaluate_batch(triple_batch, embeddings, type_to_ids_table, id_to_type_table, data.relation_count)

        # Gradient Descent
        with tf.name_scope('learn'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            lr_decay = tf.train.inverse_time_decay(FLAGS.learning_rate, global_step,
                                                   decay_steps=FLAGS.learning_decay_steps * batch_count,
                                                   decay_rate=FLAGS.learning_decay_rate)
            tf.summary.scalar('learning_rate', lr_decay)
            optimizer = tf.train.GradientDescentOptimizer(lr_decay).minimize(loss, global_step)

    # Validation
    with tf.name_scope('validation'):
        valid = tf.train.shuffle_batch([data.validation_triples], FLAGS.batch_size,
                                       capacity=2*FLAGS.batch_size, min_after_dequeue=FLAGS.batch_size,
                                       allow_smaller_final_batch=False, name='shuffle_batch')
        valid_loss = evaluate_batch(valid, embeddings, type_to_ids_table, id_to_type_table, data.relation_count)
        valid_loss_mean = tf.reduce_mean(valid_loss)

    summaries = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
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
            print('Populating id_to_type table...')
            feed_dict = {id_to_type_keys: np.array(list(data.id_to_type.keys())),
                         id_to_type_values: np.array(list(data.id_to_type.values()))}
            sess.run([id_to_type_insert], feed_dict)

            epoch = 0
            pocket_loss = 2.
            while True:
                epoch += 1

                print('Initializing projector...')
                projector_config = projector.ProjectorConfig()
                embeddings_config = projector_config.embeddings.add()
                embeddings_config.tensor_name = embeddings.name
                projector.visualize_embeddings(summary_writer, projector_config)

                print('Training epoch {}...'.format(epoch))
                for batch in range(1, batch_count):
                    # Shuffle the available corrupt entity ids every batch
                    # TODO: only use entities with metadata column isTail=true
                    padded_values = np.array([[random.choice(v) for _ in range(FLAGS.padded_size)]
                                              for v in data.type_to_ids.values()])
                    feed_dict = {type_to_ids_keys: np.array(list(data.type_to_ids.keys())),
                                 type_to_ids_values: np.array(padded_values)}
                    sess.run([type_to_ids_insert], feed_dict)

                    # Run validation and log to summary_writer
                    # TODO: this should run the entire validation set
                    if batch % (batch_count / 16) == 0:
                        vlm, summary, step = sess.run([valid_loss_mean, summaries, global_step])
                        summary_writer.add_summary(summary, step)
                        print('\tStep {} Validation Loss: {}...'.format(step, vlm))

                        # Checkpoint
                        if vlm < pocket_loss:
                            pocket_loss = vlm
                            saver.save(sess, FLAGS.output_dir + '/model.ckpt')
                            print('Epoch {}, (Model saved with loss {})'.format(epoch, vlm))

                    sess.run([optimizer])

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            print('Stopping training...')
            coord.request_stop()

        coord.join(threads)


class HolEInferenceData(HolEData):
    def __init__(self):
        self.id_to_metadata = dict()
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
            index, diffbot_id, name, diffbot_type, mentions, is_tail = line.strip().split('\t')
            index = int(index)
            mentions = int(mentions)
            if mentions >= FLAGS.min_mentions or diffbot_id.startswith('P'):
                data.type_to_ids[diffbot_type].append(index)
            data.id_to_metadata[index] = diffbot_id + ' ' + name

    data.relation_count = sum(1 for line in open(relation_file))

    print('Types: ', {k: len(v) for k, v in data.type_to_ids.items()})

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


def eval_link_prediction(scores, id_to_metadata, true_triples, test_triples, max_triples,
                         raw_positions, filtered_positions):
    heap = []
    min_loss = 100
    for pair in scores:
        loss = pair[0][0]
        min_loss = min(min_loss, loss)
        heappush(heap, (loss, tuple(pair[1])))
        person_id = id_to_metadata[pair[1][0]]
        relation = id_to_metadata[pair[1][2]]

    is_confident = min_loss < FLAGS.infer_threshold
    if is_confident:
        print('https://diffbot.com/entity/' + person_id, relation)

    raw_rank = 0
    filtered_rank = 0

    with open('inference_results.tsv', 'a') as output:
        while heap:
            pair = heappop(heap)
            loss = pair[0]
            head_id = pair[1][0]
            tail_id = pair[1][1]
            relation_id = pair[1][2]

            raw_rank += 1
            in_sample = tail_id in true_triples[head_id][relation_id]
            if is_confident and filtered_rank < max_triples:
                output.write('{:.6f}\t{}\t{}\t{}\t{}\n'.format(loss, head_id, tail_id, relation_id, in_sample))

            if is_confident and in_sample:
                print('\tTRAIN {}: {}\thttps://diffbot.com/entity/{}'.format(
                    raw_rank, loss, id_to_metadata[tail_id]))
                continue

            filtered_rank += 1
            if is_confident and tail_id in test_triples[head_id][relation_id]:
                raw_positions.append(raw_rank)
                filtered_positions.append(filtered_rank)
                print('\tMATCH {}: {}\thttps://diffbot.com/entity/{}'.format(
                    filtered_rank, loss, id_to_metadata[tail_id]))
                continue
            elif is_confident and filtered_rank <= 3:
                print('\tGUESS {}: {}\thttps://diffbot.com/entity/{}'.format(
                    filtered_rank, loss, id_to_metadata[tail_id]))


def score_mrr(raw_positions, filtered_positions):
    # TODO: push these calculations into the graph, refactor for use in training validation
    raw_positions = np.array(raw_positions)
    raw_mrr = np.mean(1.0 / raw_positions)
    mean_raw_pos = np.mean(raw_positions)

    filtered_positions = np.array(filtered_positions)
    filtered_mrr = np.mean(1.0 / filtered_positions)
    mean_filtered_pos = np.mean(filtered_positions)

    hits1 = np.mean(filtered_positions <= 1).sum() * 100
    hits3 = np.mean(filtered_positions <= 3).sum() * 100
    hits10 = np.mean(filtered_positions <= 10).sum() * 100
    print('\n\n\nRaw MRR: {} (mean position: {})'.format(raw_mrr, mean_raw_pos))
    print('Filtered MRR: {} (mean position: {})'.format(filtered_mrr, mean_filtered_pos))
    print('Hits at 1: {}, 3: {}, 10: {}'.format(hits1, hits3, hits10))


class InferenceCandidates(object):
    def __init__(self, relations, tail_candidates, max_triples, min_confidence):
        self.relations = relations
        self.tail_candidates = tail_candidates
        self.max_triples = max_triples
        self.min_confidence = min_confidence


def save_embeddings():
    data = init_inference_data()

    with tf.name_scope('save_embeddings'):
        embeddings = tf.get_variable('embeddings', [data.entity_count, FLAGS.embedding_dim],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, FLAGS.output_dir + '/model.ckpt')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                for entity in xrange(data.entity_count):
                    get_embedding_op = get_embedding('h', entity, embeddings)
                    embedding = sess.run([get_embedding_op])
                    diffbot_id = data.id_to_metadata[entity]
                    print(diffbot_id, embedding)

            except tf.errors.OutOfRangeError:
                print('Done -- entity limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


def infer_triples():
    data = init_inference_data()

    # TODO: get candidate tail type from training triples
    candidate_heads = data.type_to_ids['P']
    candidates = [
                  # InferenceCandidates([1], data.type_to_ids['1'], 2, FLAGS.infer_threshold),  # Gender
                  # InferenceCandidates([2], data.type_to_ids['2'], 3, FLAGS.infer_threshold),  # Age
                  # InferenceCandidates([6], data.type_to_ids['R'], 5, FLAGS.infer_threshold),  # Role
                  InferenceCandidates([9], [95], 1, FLAGS.infer_threshold)  # Skill, programming languages,
                  # InferenceCandidates([12, 13, 14, 15, 16], data.type_to_ids['A'], 3, FLAGS.infer_threshold)  # Location
                  ]

    with tf.name_scope('inference'):
        embeddings = tf.get_variable('embeddings', [data.entity_count, FLAGS.embedding_dim],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        triple_batch = tf.placeholder(tf.int64, [None, 3], 'triples')
        eval_loss = evaluate_triples(triple_batch, embeddings)

        # Load embeddings
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, FLAGS.output_dir + '/model.ckpt')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                raw_positions = []
                filtered_positions = []

                for head in candidate_heads:
                    for candidate in candidates:
                        candidate_relations = np.array(list(itertools.product([head], candidate.tail_candidates,
                                                                              candidate.relations)))
                        feed_dict = {triple_batch: candidate_relations}
                        triples, batch_loss = sess.run([triple_batch, eval_loss], feed_dict)

                        eval_link_prediction(zip(batch_loss, triples), data.id_to_metadata,
                                             data.true_triples, data.test_triples, candidate.max_triples,
                                             raw_positions, filtered_positions)

                score_mrr(raw_positions, filtered_positions)

            except tf.errors.OutOfRangeError:
                print('Done evaluation -- triple limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


def main(_):
    # TODO: refactor model in to object
    if FLAGS.save_embeddings:
        save_embeddings()
    elif FLAGS.infer:
        infer_triples()
    else:
        training_data = init_data()
        # TODO: param search
        run_training(training_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--learning_decay_steps', type=float, default=32, help='Learning rate decay steps (in epochs).')
    parser.add_argument('--learning_decay_rate', type=float, default=0.5, help='Learning decay rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--log_loss', action='store_true', help='Use logistic loss istead of pairwise ranking loss.')
    parser.add_argument('--l2_regularization', type=float, default=0.1, help='L2 regularization weight (log loss only).')
    parser.add_argument('--negative_ratio', type=int, default=1, help='Number of negative labels sampled in log_loss.')
    parser.add_argument('--margin', type=float, default=0.2, help='Hinge loss margin.')
    parser.add_argument('--padded_size', type=int, default=1024,
                        help='The maximum number of entities to use for each type while sampling corrupt triples.')
    parser.add_argument('--output_dir', type=str, required=True, help='Tensorboard Summary directory.')
    parser.add_argument('--data_dir', type=str, required=True, help='Input data directory.')
    parser.add_argument('--reader_threads', type=int, default=4, help='Number of training triple file readers.')
    parser.add_argument('--resume_checkpoint', action='store_true', help='Resume training on the checkpoint model.')
    parser.add_argument('--save_embeddings', action='store_true', help='Output the embeddings to .')
    parser.add_argument('--infer', action='store_true', help='Infer new triples from the latest checkpoint model.')
    parser.add_argument('--infer_threshold', type=float, default=0.05, help='Max loss to save triples')
    parser.add_argument('--min_mentions', type=int, default=50000,
                        help='The minimum number of mentions for an entity to be a viable candidate in inference.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

