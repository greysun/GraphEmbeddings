"""
Tensorflow implementation of Nickel et al, HolE, 2016

See https://arxiv.org/pdf/1510.04935.pdf
"""
import argparse
import errno
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug


FLAGS = None


def get_the_data():
    entity_file = 'diffbot_data/entity_metadata.tsv'
    relation_file = 'diffbot_data/relation_ids.txt'
    corrupt_triple_file = 'diffbot_data/triples.txt'

    entity_count = sum(1 for line in open(entity_file))
    relation_count = sum(1 for line in open(relation_file))
    triple_count = sum(1 for line in open(corrupt_triple_file))
    print "Entities: ", entity_count, "Relations: ", relation_count, "Triples: ", triple_count

    with tf.name_scope('input'):
        # Load triples from triple_file TSV
        reader = tf.TextLineReader()
        filename_queue = tf.train.string_input_producer([corrupt_triple_file] * FLAGS.num_epochs)
        key, value = reader.read(filename_queue)

        column_defaults = [tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32)]

        head_ids, tail_ids, relation_ids = tf.decode_csv(value, column_defaults, field_delim='\t')
        triples = tf.stack([head_ids, tail_ids, relation_ids])

        return entity_count, relation_count, triple_count, triples


def corrupt_heads(entity_count, relation_count, triples):
    # TODO lookup entityIds by type
    tail_column = tf.slice(triples, [0, 1], [-1, 1])
    relation_column = tf.slice(triples, [0, 2], [-1, 1])
    corrupt_head_column = tf.random_uniform([FLAGS.batch_size, 1], maxval=entity_count-relation_count, dtype=tf.int32)
    concat = tf.concat([corrupt_head_column, tail_column, relation_column], 1)
    return concat


def corrupt_tails(entity_count, relation_count, triples):
    # TODO lookup entityIds by type
    head_column = tf.slice(triples, [0, 0], [-1, 1])
    relation_column = tf.slice(triples, [0, 2], [-1, 1])
    corrupt_tail_column = tf.random_uniform([FLAGS.batch_size, 1], maxval=entity_count-relation_count, dtype=tf.int32)
    concat = tf.concat([head_column, corrupt_tail_column, relation_column], 1)
    return concat


def corrupt_entities(entity_count, relation_count, triples):
    should_corrupt_heads = tf.less(tf.random_uniform([], 0, 1.0), .5)
    return tf.cond(should_corrupt_heads,
                   lambda: corrupt_heads(entity_count, relation_count, triples),
                   lambda: corrupt_tails(entity_count, relation_count, triples))


def corrupt_relations(relation_count, triples):
    entity_columns = tf.slice(triples, [0, 0], [-1, 2])
    corrupt_relation_column = tf.random_uniform([FLAGS.batch_size, 1], maxval=relation_count, dtype=tf.int32)
    return tf.concat([entity_columns, corrupt_relation_column], 1)


def corrupt_batch(entity_count, relation_count, triples):
    should_corrupt_relation = tf.less(tf.random_uniform([], 0, 1.0), .5)
    return tf.cond(should_corrupt_relation,
                   lambda: corrupt_relations(relation_count, triples),
                   lambda: corrupt_entities(entity_count, relation_count, triples))


def circular_correlation(h, t):
    if FLAGS.cpu:
        # For prototyping only, L = tanh(relation * (head - tail)^T)
        # In other words, minimize (head -> tail) being anti-parallel to relation
        return h - t

    # these ops are GPU only!
    with tf.device('/gpu:0'):
        return tf.ifft(tf.multiply(tf.conj(tf.fft(h)), tf.fft(t)))


def init_embedding(projector_config, name, entity_count, embedding_dim):
    embedding = tf.get_variable(name, [entity_count, 2 * embedding_dim],
                                initializer=tf.random_normal_initializer)

    embeddings_config = projector_config.embeddings.add()
    embeddings_config.tensor_name = name

    return embedding


def get_embedding(layer_name, entity_ids, embeddings, embedding_dim):
    with tf.device('/cpu:0'):
        entity_embeddings = tf.nn.embedding_lookup(embeddings, entity_ids, max_norm=1)
        reshaped = tf.reshape(entity_embeddings, [-1, 2 * embedding_dim])
        return tf.complex(tf.slice(reshaped, [0, 0], [-1, embedding_dim]),
                          tf.slice(reshaped, [0, embedding_dim], [-1, embedding_dim]),
                          name=layer_name)


def complex_tanh(complex_tensor):
    summed = tf.reduce_sum(tf.real(complex_tensor) + tf.imag(complex_tensor), 1)
    return tf.tanh(summed)


def evaluate_triples(triple_batch, embeddings, embedding_dim, relation_count):
    # Load embeddings
    pos_h = tf.slice(triple_batch, [0, 0], [-1, 1], name='h_id') + relation_count
    head = get_embedding('h', pos_h, embeddings, embedding_dim)
    pos_t = tf.slice(triple_batch, [0, 1], [-1, 1], name='t_id') + relation_count
    tail = get_embedding('t', pos_t, embeddings, embedding_dim)
    pos_r = tf.slice(triple_batch, [0, 2], [-1, 1], name='r_id')
    relation = get_embedding('r', pos_r, embeddings, embedding_dim)

    # Compute loss
    with tf.name_scope('eval'):
        loss = complex_tanh(tf.matmul(relation,
                                      circular_correlation(head, tail),
                                      transpose_b=True))
        variable_summaries(loss)
        loss_scalar = tf.reduce_sum(loss)
        tf.summary.scalar('total', loss_scalar)

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


def run_training(entity_count, relation_count, triple_count, triples):
    # Initialize parameters
    margin = FLAGS.margin
    embedding_dim = FLAGS.embedding_dim
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    batch_count = triple_count / batch_size
    print "Embedding dimension: ", embedding_dim, "Batch size: ", batch_size, "Batch count: ", batch_count

    # Warning: this will clobber existing summaries
    if not FLAGS.resume_checkpoint and os.path.isdir(FLAGS.output_dir):
        shutil.rmtree(FLAGS.output_dir)
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Initialize embeddings (TF doesn't support complex embeddings, split real part and imaginary part)
    projector_config = projector.ProjectorConfig()
    embeddings = init_embedding(projector_config, 'embeddings', entity_count, embedding_dim)

    with tf.name_scope('batch'):
        # Sample triples
        triple_batch = tf.train.shuffle_batch([triples], batch_size,
                                              num_threads=FLAGS.reader_threads,
                                              capacity=2*FLAGS.reader_threads*batch_size,
                                              min_after_dequeue=batch_size,
                                              allow_smaller_final_batch=False)

        # Evaluate triples
        with tf.name_scope('train'):
            train_loss = evaluate_triples(triple_batch, embeddings, embedding_dim, relation_count)
        with (tf.name_scope('corrupt')):
            corrupt_triples = corrupt_batch(entity_count, relation_count, triple_batch)
            corrupt_loss = evaluate_triples(corrupt_triples, embeddings, embedding_dim, relation_count)

        # Score and minimize hinge-loss
        loss = tf.maximum(train_loss - corrupt_loss + margin, 0)
        # TODO: experiment with other optimizers
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Log the total batch loss
        total_loss = tf.reduce_sum(loss)
        tf.summary.scalar('loss', total_loss)

    summaries = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Save embeddings
    saver = tf.train.Saver({'embeddings': embeddings})

    supervisor = tf.train.Supervisor(logdir=FLAGS.output_dir)
    with supervisor.managed_session() as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        summary_writer = tf.summary.FileWriter(FLAGS.output_dir + '/graph', sess.graph)
        projector.visualize_embeddings(summary_writer, projector_config)

        # Load the previous model if resume_checkpoint=True
        sess.run(init_op)
        if FLAGS.resume_checkpoint:
            saver.restore(sess, FLAGS.output_dir + '/model.ckpt')
            # TODO: continue counting from last epoch

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            epoch = 0
            while not supervisor.should_stop():
                epoch += 1
                batch_losses = []
                for batch in range(1, batch_count):
                    # Train and log batch to summary_writer
                    if batch % 1000 == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, batch_loss, summary = sess.run([optimizer, total_loss, summaries],
                                                          options=run_options,
                                                          run_metadata=run_metadata)
                        step = '{}-{}'.format(epoch, batch)
                        summary_writer.add_run_metadata(run_metadata, step)
                        supervisor.summary_computed(sess, summary)
                        print "\tSaved summary for step " + step
                    else:
                        _, batch_loss = sess.run([optimizer, total_loss])
                    batch_losses.append(batch_loss)

                # Checkpoint
                save_path = saver.save(sess, FLAGS.output_dir + '/model.ckpt', epoch)
                print('Epoch {} Loss: {} (Model saved as {})'.format(epoch, np.mean(batch_losses), save_path))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


def main(_):
    entity_count, relation_count, triple_count, triples = get_the_data()
    run_training(entity_count, relation_count, triple_count, triples)


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
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=64,
        help='Embedding dimension.'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.,
        help='Hinge loss margin.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='holE',
        help='Tensorboard Summary directory.'
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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

