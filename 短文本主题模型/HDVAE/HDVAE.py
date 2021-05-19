import numpy as np
import tensorflow as tf
import math
import os
import utils as utils
import sys
import argparse
import pickle
import random
import pdb
os.environ["CUDA_VISIBLE_DEVICES"]="0"

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('n_hidden', 256, 'Size of each hidden layer.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'sigmoid', 'Non-linearity of the MLP.')
flags.DEFINE_string('summaries_dir','summaries','where to save the summaries')
FLAGS = flags.FLAGS

def parseArgs():
    global dataset_name, args
    dataset_name= "agnews"
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--adam_beta1',default=0.9, type=float)
    argparser.add_argument('--adam_beta2',default=0.999, type=float)
    argparser.add_argument('--learning_rate',default=1e-3, type=float)
    argparser.add_argument('--dir_prior',default=0.1, type=float)
    argparser.add_argument('--n_topic',default=50, type=int)
    argparser.add_argument('--n_sample',default=1, type=int)
    argparser.add_argument('--warm_up_period',default=20, type=int)
    argparser.add_argument('--data_dir',default='../data/'+dataset_name, type=str) #  数据集
    args = argparser.parse_args()
    return args




class DirVAE(object):

    def calculate_kld(self, mask, alpha, prior):
        kld = tf.lgamma(tf.reduce_sum(alpha,axis=1))-tf.lgamma(tf.reduce_sum(prior,axis=1))
        kld-=tf.reduce_sum(tf.lgamma(alpha),axis=1)
        kld+=tf.reduce_sum(tf.lgamma(prior),axis=1)
        minus = alpha-prior
        test = tf.reduce_sum(tf.multiply(minus,tf.digamma(alpha)-tf.reshape(tf.digamma(tf.reduce_sum(alpha,1)),(self.batch_size,1))),1)
        kld+=test
        kld = mask*kld  # mask paddings
        return kld




    def __init__(self, 
                 vocab_size,
                 n_hidden,
                 n_topic,
                 learning_rate,
                 batch_size,
                 non_linearity,
                 adam_beta1,
                 adam_beta2,
                 dir_prior):
        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = 1 #n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        lda=False
        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.warm_up = tf.placeholder(tf.float32, (), name='warm_up')  # warm up
        self.adam_beta1=adam_beta1
        self.adam_beta2=adam_beta2
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.min_alpha = tf.placeholder(tf.float32,(), name='min_alpha')

        

        with tf.variable_scope('encoder'):
            self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity, scope="Linear1")
            self.index = utils.mlp(self.x, [5], tf.nn.softmax, scope="Index1")
            self.enc_vec = tf.nn.dropout(self.enc_vec,self.keep_prob)
            self.mean1 = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean1'))
            self.alpha1 = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(self.mean1)))
            # self.alpha = self.alpha + 1
            self.mean2 = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean2'))
            self.alpha2 = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(self.mean2)))

            self.mean3 = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean3'))
            self.alpha3 = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(self.mean3)),)

            self.mean4 = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean4'))
            self.alpha4 = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(self.mean4)),)

            self.mean5 = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean5'))
            self.alpha5 = tf.maximum(self.min_alpha,tf.log(1.+tf.exp(self.mean5)),)


            self.prior1 = tf.ones((batch_size,self.n_topic), dtype=tf.float32, name='prior1')*0.1
            self.prior2 = tf.ones((batch_size,self.n_topic), dtype=tf.float32, name='prior2')*0.3
            self.prior3 = tf.ones((batch_size,self.n_topic), dtype=tf.float32, name='prior3')*0.5
            self.prior4 = tf.ones((batch_size,self.n_topic), dtype=tf.float32, name='prior4')*0.7
            self.prior5 = tf.ones((batch_size,self.n_topic), dtype=tf.float32, name='prior5')*0.9

            
            self.kld1 = self.calculate_kld(self.mask, self.alpha1, self.prior1)
            self.kld2 = self.calculate_kld(self.mask, self.alpha2, self.prior2)
            self.kld3 = self.calculate_kld(self.mask, self.alpha3, self.prior3)
            self.kld4 = self.calculate_kld(self.mask, self.alpha4, self.prior4)
            self.kld5 = self.calculate_kld(self.mask, self.alpha5, self.prior5)
            self.alpha = tf.expand_dims(self.index[:, 0], 1) * self.alpha1 + tf.expand_dims(self.index[:, 1], 1) * self.alpha2 + tf.expand_dims(self.index[:, 2],1) * self.alpha3 + \
                        tf.expand_dims(self.index[:, 3],1) * self.alpha4 + tf.expand_dims(self.index[:, 4],1) * self.alpha5
            self.kld = self.index[:, 0] * self.kld1 + self.index[:, 1] * self.kld2 + self.index[:, 2] * self.kld3 + self.index[:, 3] * self.kld4 + self.index[:, 4] * self.kld5






        B = 0
        with tf.variable_scope('decoder'):
            if self.n_sample ==1:
                # 逆CDF
                # self.u = tf.random_uniform((batch_size,self.n_topic))
                
                # self.alpha = self.alpha + tf.to_float(10)
                
                gam = tf.squeeze(tf.random_gamma(shape = (1,),alpha=self.alpha+B, beta=1))
                self.u = tf.minimum(1., tf.stop_gradient(tf.pow(gam, self.alpha+B-1) * tf.exp(-gam) 
                            / (tf.exp(tf.lgamma(self.alpha+B)))))
                self.doc_vec = tf.pow(self.u*(self.alpha+B)*tf.exp(tf.lgamma(self.alpha+B)),
                                1./(self.alpha+B))
                # pdb.set_trace()
                self.doc_vec = tf.div(self.doc_vec,tf.reshape(tf.reduce_sum(self.doc_vec,1), (-1, 1)))
                # self.doc_vec = tf.div(gam,tf.reshape(tf.reduce_sum(gam,1), (-1, 1)))
                self.doc_vec.set_shape(self.alpha.get_shape())
                # self.theta = self.doc_vec

                # 拒绝采样
                # e = tf.random_normal((batch_size, self.n_topic))
                # u = tf.random_uniform((batch_size, self.n_topic))
                # with tf.variable_scope('prob'):
                #     mask = tf.cast(tf.less(self.alpha, 1), dtype=tf.float32)
                #     self.tmp = self.alpha + mask # 令小于1的alpha加1
                #     self.doc_vec = (self.tmp-1./3) * tf.pow((1 + e / tf.sqrt(9*self.tmp-3)), 3)
                #     reshape = tf.pow(u, 1./self.alpha)*mask + tf.cast(tf.greater_equal(self.alpha, 1), dtype=tf.float32)
                #     self.doc_vec *= reshape # 仅对加了1的alpha对应的样本进行reshape

                # gsm 方法
                # topic_vec = tf.Variable(tf.glorot_uniform_initializer()((self.vocab_size, self.n_hidden)))
                # word_vec = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic, self.n_hidden)))
                # beta = tf.nn.softmax(tf.matmul(topic_vec, word_vec, transpose_b=True), axis=0)  # (V, K)
                # self.beta = tf.transpose(beta)                
                # logits = tf.log(tf.matmul(self.theta, self.beta))
                # self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)

                # nvdm方法
                if lda:
                    logits = tf.log(tf.clip_by_value(utils.linear_LDA(self.doc_vec, self.vocab_size, scope='projection',no_bias=True),1e-10,1.0))
                else:
                    logits = tf.nn.log_softmax(tf.contrib.layers.batch_norm(utils.linear(self.doc_vec, self.vocab_size, scope='projection',no_bias=True)))
                self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)

                # copula
                self.lambd = tf.constant(shape=[self.batch_size, 1], value=1.)
                self.log_prob = (-self.n_topic-(1./self.lambd)) * tf.log( # 加上1e-12，防止求梯度出现除0
                            tf.reduce_sum(tf.pow(self.doc_vec+1e-12, -self.lambd), -1, keepdims=True) - self.n_topic + 1)
                constant_term = 0.0
                for i in range(self.n_topic):
                    constant_term += tf.log(1 + i*self.lambd)
                self.log_prob += constant_term
                self.log_prob *= self.mask
                self.log_prob += 300
                self.log_prob = tf.clip_by_value(self.log_prob, 0, np.inf)

        self.objective = self.recons_loss + self.kld
        self.loss = self.objective# - 0.03*self.log_prob

        self.fullvars = tf.trainable_variables()
        self.enc_vars = utils.variable_parser(self.fullvars, 'encoder')
        self.dec_vars = utils.variable_parser(self.fullvars, 'decoder')

        self.dec_grads = tf.gradients(self.loss, self.dec_vars)
        kl_grad = tf.gradients(self.kld, self.enc_vars)
        g_rep = tf.gradients(self.recons_loss, self.enc_vars)
        self.enc_grads = [g_r + self.warm_up * g_k for g_r, g_k in zip(g_rep, kl_grad)]





        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.adam_beta1,beta2=self.adam_beta2)
        self.optim_enc = optimizer.apply_gradients(zip(self.enc_grads, self.enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(self.dec_grads, self.dec_vars))
        # self.optim_all = optimizer.apply_gradients(zip(self.fullgrads, self.fullvars))
        self.optim_all = optimizer.apply_gradients(list(zip(self.enc_grads, self.enc_vars)) +
                                                    list(zip(self.dec_grads, self.dec_vars)))


def train(sess, model,
          train_url,
          test_url,
          batch_size,
          vocab_size,
          alternate_epochs=1,
          lexicon=[],
          result_file='test.txt',
          warm_up_period=100):
    train_set, train_count = utils.data_set(train_url)
    test_set, test_count = utils.data_set(test_url)
    #%%
    train_size=len(train_set)
    print("train_size = ", train_size)
    validation_size=int(train_size*0.1)
    dev_set = train_set[:validation_size]
    dev_count = train_count[:validation_size]
    # train_set = train_set[validation_size:]
    # train_count = train_count[validation_size:]

    optimize_jointly = True
    train_batches_not_shuffle = utils.create_batches(len(train_set), batch_size, shuffle=False)
    dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)
    warm_up = 0
    min_alpha = 0.00001

    best_print_ppx=1e10
    early_stopping_iters=1000
    no_improvement_iters=0
    stopped=False
    epoch=0
    switch = 0
    train_whole_record = []
    test_whole_record = []
    npmi_whole = []
    tu_whole = []
    while not stopped:
        epoch+=1
        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
        if warm_up<1.:
            warm_up += 1./warm_up_period
        else:
            warm_up=1.

        if optimize_jointly:
            optim = model.optim_all
            print_mode = 'updating encoder and decoder'
        elif switch == 0:
            optim = model.optim_dec
            print_mode = 'updating decoder'
            switch = 1
        else:
            optim = model.optim_enc
            print_mode = 'updating encoder'
            switch = 0


        
        for i in range(alternate_epochs):
            loss_sum = 0.0
            ppx_sum = 0.0
            kld_sum = 0.0
            word_count = 0
            doc_count = 0
            recon_sum=0.0
            count = 0
            train_theta = []
            for idx_batch in train_batches:
                data_batch, count_batch, mask = utils.fetch_data(train_set, train_count, idx_batch, vocab_size)
                input_feed = {model.x: data_batch, model.mask: mask, model.keep_prob: 0.75, model.warm_up: warm_up, model.min_alpha: min_alpha}
                doc_vec, u1, alpha = sess.run((model.doc_vec, model.u, model.alpha), input_feed)
                _, loss, objective, recon, kld, log_prob, alpha, doc_vec = \
                             sess.run((optim, model.loss, model.objective, model.recons_loss, model.kld, model.log_prob, model.alpha, model.doc_vec), input_feed)
                
                
                loss_sum += np.sum(objective)
                kld_sum += np.sum(kld) / np.sum(mask)
                word_count += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)
                ppx_sum += np.sum(np.divide(objective, count_batch))
                doc_count += np.sum(mask)
                recon_sum+=np.sum(recon)
                train_theta.extend(doc_vec)
                # import pdb
                # pdb.set_trace()
            print("1111 len(train_theta) ", len(train_theta))
#             print(t3)
#             print(t4[0])
#             print(t5[0])
#             print(alpha)
#             print(t6)
#             print(t3*t6)
#             print(np.sum(t3*t6, 1))
            print_ppx = np.exp(loss_sum / word_count)
            print_ppx_perdoc = np.exp(ppx_sum / doc_count)
            print_kld = kld_sum/len(train_batches)
            print_recon = recon_sum/doc_count
            print('| Epoch train: {:d} |'.format(epoch+1),
#                 print_mode, '{:d}'.format(i),
               '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
               '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
               '| KLD: {:.5f}'.format(print_kld),
               '| Rec: {:.5f}'.format(print_recon))
        train_whole_record.append([epoch, print_ppx, print_ppx_perdoc, 
                                    print_kld, print_recon])

        loss_sum = 0.0
        ppx_sum = 0.0
        kld_sum = 0.0
        word_count = 0
        doc_count = 0
        recon_sum=0.0
        for idx_batch in dev_batches:
            data_batch, count_batch, mask = utils.fetch_data(dev_set, dev_count, idx_batch, vocab_size)
            input_feed = {model.x: data_batch, model.mask: mask, model.keep_prob: 1.0, model.warm_up: 1.0, model.min_alpha: min_alpha}
            loss, objective, recon, kld = sess.run((model.loss,
                                        model.objective, model.recons_loss, model.kld), input_feed)
            loss_sum += np.sum(objective)
            kld_sum += np.sum(kld) / np.sum(mask)
            word_count += np.sum(count_batch)
            count_batch = np.add(count_batch, 1e-12)
            ppx_sum += np.sum(np.divide(objective, count_batch))
            doc_count += np.sum(mask)
            recon_sum+=np.sum(recon)

        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_kld = kld_sum/len(dev_batches)

        if print_ppx<best_print_ppx:
            no_improvement_iters=0
            best_print_ppx=print_ppx
            tf.train.Saver().save(sess, 'models1/improved_model_rcdf')
        else:
            no_improvement_iters+=1
            # print('no_improvement_iters',no_improvement_iters,'best ppx',best_print_ppx)
            if no_improvement_iters>=early_stopping_iters:
                stopped=True
                print('stop training after',epoch,'iterations,no_improvement_iters',no_improvement_iters)
                print('load stored model')
                tf.train.Saver().restore(sess,'models1/improved_model_rcdf')
        # print('| Epoch dev: {:d} |'.format(epoch+1),
        #       '| Corpus ppx: {:.9f}'.format(print_ppx),
        #       '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
        #       '| KLD: {:.5}'.format(print_kld))

        if FLAGS.test :
            loss_sum = 0.0
            ppx_sum = 0.0
            kld_sum = 0.0
            word_count = 0
            doc_count = 0
            recon_sum = 0.0
            test_theta = []
            for idx_batch in test_batches:
                data_batch, count_batch, mask = utils.fetch_data(test_set, test_count, idx_batch, vocab_size)
                input_feed = {model.x: data_batch, model.mask: mask, model.keep_prob: 1.0, model.warm_up: 1.0, model.min_alpha: min_alpha}
                loss, objective, recon, kld, doc_vec = \
                                            sess.run((model.loss,
                                                model.objective, model.recons_loss, model.kld, model.doc_vec), input_feed)
                loss_sum += np.sum(objective)
                kld_sum += np.sum(kld) / np.sum(mask)
                word_count += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)
                ppx_sum += np.sum(np.divide(objective, count_batch))
                doc_count += np.sum(mask)
                recon_sum+=np.sum(recon)
                test_theta.extend(doc_vec)

            print_ppx = np.exp(loss_sum / word_count)
            print_ppx_perdoc = np.exp(ppx_sum / doc_count)
            print_kld = kld_sum/len(test_batches)
            print_recon = recon_sum/doc_count
            print('| Epoch test: {:d} |'.format(epoch+1), 
                  '| Corpus ppx: {:.9f}'.format(print_ppx),
                  '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
                  '| KLD: {:.5}'.format(print_kld),
                  '| Rec: {:.5}'.format(print_recon))
            test_whole_record.append([epoch, print_ppx, print_ppx_perdoc, 
                                    print_kld, print_recon])
            
            
            doc_word_data = []
            for idx_batch in test_batches:
                data_batch, _, _ = utils.fetch_data(
                test_set, test_count, idx_batch, vocab_size)
                doc_word_data.extend(data_batch)
            for idx_batch in train_batches:
                data_batch, _, _ = utils.fetch_data(
                train_set, train_count, idx_batch, vocab_size)
                doc_word_data.extend(data_batch)
            doc_word_data = np.array(doc_word_data)
            dec_vars = utils.variable_parser(tf.trainable_variables(), 'decoder')
            phi = dec_vars[0]
            phi = sess.run(phi)
            # gsm
            # phi = beta
            coherence = utils.evaluate_coherence(phi, doc_word_data, [10])
           
            utils.print_top_words(phi, lexicon,result_file=None)




            TU = utils.evaluate_TU(phi, [10])
            print('topic coherence',str(coherence))
            print("TU score", str(TU))
            tu_whole.append([epoch, TU])
            npmi_whole.append([epoch, coherence])
#                 with open(os.path.join(model_path, 'coherence_result.txt'), 'w', encoding='utf-8') as f:
#                     f.writelines('topic coherence {co}'.format(co=str(coherence)))
        if epoch % 1 == 0:
            

            train_theta = []
            for idx_batch in train_batches_not_shuffle:
                data_batch, count_batch, mask = utils.fetch_data(train_set, train_count, idx_batch, vocab_size)
                input_feed = {model.x: data_batch, model.mask: mask, model.keep_prob: 0.75, model.warm_up: warm_up, model.min_alpha: min_alpha}
                doc_vec = sess.run((model.doc_vec), input_feed)
                train_theta.extend(doc_vec)
            print("len(train_theta ", len(train_theta))


            dir_name = os.path.join("result", dataset_name, str(args.n_topic))
            utils.save_middle_var(train_theta, os.path.join(dir_name, "theta", str(epoch)+"_train-theta.pkl"))
            utils.save_middle_var(test_theta, os.path.join(dir_name, "theta", str(epoch)+"_test-theta.pkl"))
            # dir_name = os.path.join("result", dataset_name, , "topic_word")
            utils.save_middle_var(phi, os.path.join(dir_name, "topic_word", str(epoch)+".pkl"))
        # if epoch == 300:
        #     break
    dir_name = os.path.join("result", dataset_name, str(args.n_topic), "record")
    utils.save_middle_var(train_whole_record, os.path.join(dir_name, "train-record.pkl"))
    utils.save_middle_var(test_whole_record, os.path.join(dir_name, "test-record.pkl"))
    utils.save_middle_var(tu_whole, os.path.join(dir_name, "tu.pkl"))
    utils.save_middle_var(npmi_whole, os.path.join(dir_name, "npmi.pkl"))



def myrelu(features):
    return tf.maximum(features, 0.0)

def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
        non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
        non_linearity = tf.nn.sigmoid
    else:
        non_linearity = myrelu

    args = parseArgs()
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    learning_rate = args.learning_rate
    dir_prior = args.dir_prior
    warm_up_period = args.warm_up_period
    n_sample = args.n_sample
    n_topic = args.n_topic
    lexicon=[]
    vocab_path = os.path.join(args.data_dir, 'vocab.new')
    with open(vocab_path,'r') as rf:
        for line in rf:
            word = line.split()[0]
            lexicon.append(word)
    vocab_size=len(lexicon)
    nvdm = DirVAE(vocab_size=vocab_size,
                n_hidden=FLAGS.n_hidden,
                n_topic=n_topic, 
                learning_rate=learning_rate, 
                batch_size=FLAGS.batch_size,
                non_linearity=non_linearity,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                dir_prior=dir_prior)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    result = sess.run(init)
    train_url = os.path.join(args.data_dir, 'train.feat')
    test_url = os.path.join(args.data_dir, 'test.feat')

    train(sess, nvdm, train_url, test_url, FLAGS.batch_size,vocab_size,lexicon=lexicon,
                result_file=None,
                warm_up_period = warm_up_period)

if __name__ == '__main__':
    tf.app.run()