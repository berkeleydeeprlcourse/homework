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

    def test_sum_discounted_rewards(self):
        rewards = [1, 10, 100]
        gamma = .9
        sum_discount_rewards = train_pg.sum_discounted_rewards(rewards, gamma)
        self.assertEqual(sum_discount_rewards, 1 + 9 + 81)


    def test_discounted_rewards_to_go(self):
        rewards = [1, 10, 100]
        gamma = .9
        rtgs = train_pg.discounted_rewards_to_go(rewards, gamma)
        self.assertEqual(rtgs, [91, 100, 100])

    def test_discrete_policy_network(self):
        with self.test_session() as session:
            sy_logits_na = tf.log([[2., 1.]])
            sy_logits_na_val = session.run(sy_logits_na)
            self.assertArrayNear(sy_logits_na_val[0], [0.693147, 0.0], err=1e-4)

            # draw one sample from a multinomial distribution
            logits_multinomial = tf.multinomial(sy_logits_na, 1, seed=1234)
            logits_multinomial_val = session.run(logits_multinomial)
            self.assertEqual(logits_multinomial_val, [[1]])

            sampled_ac = tf.squeeze(logits_multinomial_val, axis=[1])
            sampled_ac_val = session.run(sampled_ac)
            self.assertEqual(sampled_ac_val, [1])

            # Compute the log probability of a set of actions that were actually taken, according to the policy.
            sy_ac_na = [0]
            sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)
            sy_logprob_n_val = session.run(sy_logprob_n)
            self.assertArrayNear(sy_logprob_n_val, [0.4054651], err=1e-4)

    def test_continuous_policy_network(self):
        with self.test_session() as session:
            sy_mean = [1., -1]              
            sy_logstd = tf.log([1, 2.])
            sy_ac_na = [-1., 0]

            sampled_ac = tf.random_normal(shape=tf.shape(sy_mean), mean=sy_mean, stddev=tf.exp(sy_logstd), seed=1234) 
            sampled_ac_val = session.run(sampled_ac)
            self.assertArrayNear(sampled_ac_val, [1.5134048, -1.5116279], err=1e-4)

            dist = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_mean, 
                    scale_diag=tf.exp(sy_logstd)) 

            sy_logprob_n = -dist.log_prob(sy_ac_na).eval()
            self.assertAlmostEqual(sy_logprob_n, 4.6560245)

if __name__ == '__main__':
    tf.test.main()
