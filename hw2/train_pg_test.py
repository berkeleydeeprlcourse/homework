import tensorflow as tf
import train_pg
from tensorflow.python.platform import gfile

class TrainPGTest(tf.test.TestCase):
    def test_build_mlp(self):
        with self.test_session() as session:

            input_features = 64
            input_placeholder = tf.placeholder(shape=[64, 128], name="ob", dtype=tf.float32)
            output_size = 10
            mlp_output_layer = train_pg.build_mlp(input_placeholder, 
                output_size=output_size,
                scope='MLP',
                n_layers=3,
                size=256,
                activation=tf.tanh,
                output_activation=None)

            # tf.train.write_graph(session.graph_def, ".", "train_pg_test_graph.pb", False)

            self.assertAllEqual(mlp_output_layer.get_shape().as_list(), [input_features, output_size])

            with tf.Session() as session2:
                with gfile.FastGFile("train_pg_test_graph.pb",'rb') as f:
                    expected_graph_def = tf.GraphDef()
                    expected_graph_def.ParseFromString(f.read())
                    session2.graph.as_default()
                    tf.import_graph_def(expected_graph_def, name='')

            print(tf.get_default_graph().as_graph_def())
            tf.test.assert_equal_graph_def(tf.get_default_graph().as_graph_def(), session2.graph.as_graph_def())

if __name__ == '__main__':
    tf.test.main()
