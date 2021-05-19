import tensorflow as tf
import random
import numpy as np
from tensorflow.contrib import rnn
import pickle 
import os

def get_vocab_bow(path):
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  vocab_bow = [data.split()[0] for data in lines]
  return vocab_bow


def get_train_neighbors(train_batch_url):
  '''
  得到train_neighbors. 
  Args
    train_batch_url：
  
  Returns:
    train_neighbors: 数组，数组元素是邻居编号
  '''
  with open(train_batch_url, 'rb') as f:
    train_neighbors = pickle.load(f)
  return train_neighbors



def get_batch_neighbors(train_neighbors, idx_batch, ):
  ###-Dc
  '''
  得到每个batch里每条文本的邻居
  Args:
    train_neighbors: 训练集邻居
    idx_batch：batch的index
  
  Returns:
    batch_neighbors_mat: batch的邻居的0-1矩阵
  '''
  idx_dict = {}
  n = len(idx_batch)
  # 建立idx_batch的字典
  for i, idx in enumerate(idx_batch):
    idx_dict[idx] = i
  batch_neighbors_mat = np.zeros(shape=(n, n), dtype=np.float32)
  cnt = 0
  for i in range(n):
    idx = idx_batch[i]
    for ele in train_neighbors[idx]:
      # print("idx_dict[ele] ", idx_dict[ele])
      if ele in idx_dict.keys():
        batch_neighbors_mat[i, idx_dict[ele]] = 1.
        cnt += 1
  # print("batch_neighbors_mat", batch_neighbors_mat)
  print("cnt ", cnt)

  return batch_neighbors_mat
      
def get_batch_neighbors_ind(train_neighbors, idx_batch):
  idx_dict = {}
  n = len(idx_batch)
  m = train_neighbors.shape[1]
  # 建立idx_batch的字典
  for i, idx in enumerate(idx_batch):
    idx_dict[idx] = i
  batch_neighbors_idx = []

  total = 0
  for i in range(n):
    neighbors_idx = []
    idx = idx_batch[i]
    cnt = 0  #  记录batch里的每一个文本的邻居数
    for ele in train_neighbors[idx]:
      # print("idx_dict[ele] ", idx_dict[ele])
      if ele in idx_dict.keys():
        neighbors_idx.append(idx_dict[ele])
        cnt += 1
    # print("cnt ", cnt)
    total += cnt
    neighbors_idx.extend([i] * (m-cnt+1))
    batch_neighbors_idx.append(neighbors_idx)
  # print("batch_neighbors_mat", batch_neighbors_mat)
  # print("cnt ", cnt)
  # print("total = ", total)

  return np.array(batch_neighbors_idx)

## GSM获取数据集方式
def data_set2(data_url, vocab_size):
    """process data input."""
    data_list = []
    word_count = []
    with open(data_url) as fin:
      while True:
        line = fin.readline()
        if not line:
          break
        id_freqs = line.split()
        id_freqs = id_freqs[1:-1]
        doc = {}
        count = 0
        #print(id_freqs)
        for id_freq in id_freqs:
          items = id_freq.split(':')
          # python starts from 0
          #print(items)
          doc[int(items[0]) - 1] = int(items[1])
          count += int(items[1])
        if count > 0:
          data_list.append(doc)
          word_count.append(count)

    data_mat = np.zeros((len(data_list), vocab_size), dtype=np.float)
    for doc_idx, doc in enumerate(data_list):
      for word_idx, count in doc.items():
        data_mat[doc_idx, word_idx] += count

    return data_list, data_mat, word_count


def data_set4(data_url, vocab_size):
  """process data input."""
  print("data_url=", data_url)
  data = []
  word_count = []
  label = []
  fin = open(data_url)
  for line in fin:
    line = line.strip("\n")
    # print(line)
    # if line == "":
    #   continueten
    if line == "":
      continue
    id_freqs = line.split()
    # print(id_freqs)
    doc_label = id_freqs[0]
    doc = {}
    count = 0
    for id_freq in id_freqs[1:]:
      items = id_freq.split(':')
      # python starts from 0
      doc[int(items[0])-1] = int(items[1])
      count += int(items[1])
    if count > 0:
      data.append(doc)
      word_count.append(count)
      label.append(doc_label)
  fin.close()
  data_mat = np.zeros((len(data), vocab_size), dtype=np.float)
  for doc_idx, doc in enumerate(data):
    for word_idx, count in doc.items():
      data_mat[doc_idx, word_idx] += count


  return data, data_mat, word_count, label

import os
def fetch_seq1(data, idx_batch):
  n_step = 40
  seq = []
  for idx in idx_batch:
    idx_seq = []
    idx_seq.extend(data[idx])
    l = len(data[idx])
    if l < n_step:
      idx_seq.extend([0]*(n_step-l))
    # print(idx_seq, "|||", len(idx_seq))
    
    seq.append(idx_seq)
  seq = np.array(seq)
  # os._exit(-1)
  return seq

# 保存中间变量：如theta、ppx、tu、coh
def save_middle_var(theta, output_path):
  dir_name = os.path.dirname(output_path)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  with open(output_path, 'wb') as f:
    pickle.dump(theta, f)

def write_var_addmode(var, output_path):
  dir_name = os.path.dirname(output_path)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  with open(output_path, "a+") as f:
    f.writelines(" ".join(var))
  pass



def fetch_seq(data, idx_batch):
  n_step = 850
  seq = []
  for idx in idx_batch:
    idx_seq = []
    keys = list(data[idx].keys())
    # print(keys)
    idx_seq.extend(keys[:n_step])
    if len(keys) < n_step:
      idx_seq.extend([0]*(n_step-len(keys)))
    # print(len(idx_seq))

    seq.append(idx_seq)
  seq = np.array(seq)
  # logging.info("this is debug mess")
  # print(seq.shape)
  # os._exit(-1)
  return seq



# 根据data_batch返回对应的one_hot向量
def fetch_index(data_batch):
  row, col = data_batch.shape
  data_index = np.zeros(shape=(row, col))
  for i in range(row):
    for j in range(col):
      if data_batch[i, j] != 0:
        data_index[i, j] = j+1
  return data_index



def get_embedding_matrix(file_path, word_size):
  # 根据file_path得到embedding的矩阵
  with open(file_path, 'rb') as f:
    embedding_dict = pickle.load(f)
  # print(embedding_dict)
  embedding_matrix = []
  embedding_matrix.append([0.] * word_size)

  for k, v in embedding_dict.items():
    embedding_matrix.append(v)
  embedding_matrix = np.array(embedding_matrix, dtype=np.float32)
  return embedding_matrix

def BiLSTM(x,
           n_steps,  # 步长
           output_size,  # 输出size
           n_hidden,  # 隐藏层数量
           name=None):
  print("x.shape=", x.shape)
  lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
  lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

  outputs, states= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                              lstm_bw_cell,
                                              x,
                                              dtype=tf.float32)
  output_fw, output_bw = outputs
  print(outputs[0].shape)
  outputs = tf.concat([output_fw, output_bw], axis=-1)
  print(outputs.shape)
  with tf.variable_scope(name or 'BiLSTM'):
    weights = tf.get_variable("Weights", [2*n_hidden, output_size],)
    bias = tf.get_variable("Bias", [output_size])
  outputs = tf.matmul(outputs[:,-1,:], weights) + bias

  print("output.shape=", outputs.shape)
  return outputs



def compute_TU(topic_word, N):
  topic_size, word_size = np.shape(topic_word)
  # find top words'index of each topic
  topic_list = []
  for topic_idx in range(topic_size):
    top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
    topic_list.append(top_word_idx)
  TU= 0
  cnt =[0 for i in range(word_size)]
  for topic in topic_list:
    for word in topic:
      cnt[word]+=1
  for topic in topic_list:
    TU_t = 0
    for word in topic:
      TU_t+=1/cnt[word]
    TU_t/=N
    TU+=TU_t

  TU/=topic_size
  # compute coherence of each topic

  return TU


def evaluate_TU(topic_word,  n_list):
    #    topic_word = net.out_fc.weight.detach().t()
    #    topic_word = torch.softmax(topic_word, dim=1).detach().cpu().numpy()

    TU = 0.0
    for n in n_list:
        TU += compute_TU(topic_word, n)
    TU /= len(n_list)

    return TU



def evaluate_coherence(topic_word, doc_word, N_list):
  print('Computing coherence ...')
  topic_size = len(topic_word)
  doc_size = len(doc_word)

  average_coherence = 0.0
  for N in N_list:
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
      top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
      topic_list.append(top_word_idx)

    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
      word_array = topic_list[i]
      sum_score = 0.0
      for n in range(N):
        flag_n = doc_word[:, word_array[n]] > 0
        p_n = np.sum(flag_n) / doc_size
        for l in range(n + 1, N):
          flag_l = doc_word[:, word_array[l]] > 0
          p_l = np.sum(flag_l)
          p_nl = np.sum(flag_n * flag_l)
          # if p_n * p_l * p_nl > 0:
          if p_nl == doc_size:
            sum_score += 1
          elif p_n > 0 and p_l > 0 and p_nl > 0:
            p_l = p_l / doc_size
            p_nl = p_nl / doc_size
            sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
      sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / topic_size
    average_coherence += sum_coherence_score
  average_coherence /= len(N_list)
  return average_coherence


def data_set(data_url):
  """process data input."""
  data = []
  word_count = []
  fin = open(data_url)
  while True:
    line = fin.readline()
    if not line:
      break
    id_freqs = line.split()
    doc = {}
    count = 0
    for id_freq in id_freqs[1:]:
      items = id_freq.split(':')
      # python starts from 0
      if int(items[0])-1<0:
        print('WARNING INDICES!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      doc[int(items[0])-1] = int(items[1])
      count += int(items[1])
    if count > 0:
      data.append(doc)
      word_count.append(count)
  fin.close()
  return data, word_count

def create_batches(data_size, batch_size, shuffle=True):
  """create index by batches."""
  batches = []
  ids = list(range(data_size))
  if shuffle:
    random.shuffle(ids)
  for i in range(int(data_size / batch_size)):
    start = i * batch_size
    end = (i + 1) * batch_size
    batches.append(ids[start:end])
  # the batch of which the length is less than batch_size
  rest = data_size % batch_size
  if rest > 0:
    batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
  return batches

def fetch_data(data, count, idx_batch, vocab_size):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  data_batch = np.zeros((batch_size, vocab_size))
  count_batch = []
  mask = np.zeros(batch_size)
  indices = []
  values = []
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      for word_id, freq in data[doc_id].items():
        data_batch[i, word_id] = freq
      count_batch.append(count[doc_id])
      mask[i]=1.0
    else:
      count_batch.append(0)
  return data_batch, count_batch, mask

def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
    elif prefix in varname:
      ret_list.append(var)
  return ret_list

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def linear_LDA(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  None
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
    matrix = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.Variable(xavier_init(input_size, output_size))))
    
    output = tf.matmul(inputs, matrix)#no softmax on input, it should already be normalized
    if not no_bias:
      bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
      output = output + bias_term
  return output


def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None,
           weights=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer =  tf.truncated_normal_initializer(mean = 0.0, stddev=0.01)
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
   
    if weights is not None:
      matrix=weights
    else:
      matrix = tf.get_variable('Matrix', [input_size, output_size],initializer=matrix_initializer)
    
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
      output = output + bias_term
  return output

def mlp(inputs, 
        mlp_hidden=[], 
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    return res
    
    

def print_top_words1(topic_word_matrix, feature_names, n_top_words):
  # 打印每个主题下权重较高的term
  for topic_idx, topic in enumerate(topic_word_matrix):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
  # 打印主题-词语分布矩阵





def print_top_words(beta, feature_names, n_top_words=10,label_names=None,result_file=None):
    print('---------------Printing the Topics------------------')
    if result_file!=None:
      result_file.write('---------------Printing the Topics------------------\n')
    for i in range(len(beta)):
        topic_string = " ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        print(topic_string)
        if result_file!=None:
          result_file.write(topic_string+'\n')
    if result_file!=None:
      result_file.write('---------------End of Topics------------------\n')
    print('---------------End of Topics------------------')

def count_word_combination(dataset,combination):
  count = 0
  w1,w2 = combination
  for data in dataset:
    w1_found=False
    w2_found=False
    for word_id, freq in data.items():
      if not w1_found and word_id==w1:
        w1_found=True
      elif not w2_found and word_id==w2:
        w2_found=True
      if w1_found and w2_found:
        count+=1
        break
  return count
  
def count_word(dataset,word):
  count=0
  for data in dataset:
    for word_id, freq in data.items():
      if word_id==word:
        count+=1
        break
  return count      

def topic_coherence(dataset,beta, feature_names, n_top_words=10):
  word_counts={}
  word_combination_counts={}
  length = len(dataset)
  #go through dataset:
  #for each word combination:
    #\frac{log\frac{P(wi,wj)}{P(wi)*P(wj)}}{-logP(wi,wj)}
  coherence_sum=0.0
  coherence_count=0
  topic_coherence_sum=0.0
  
  for i in range(len(beta)):
    top_words = [j
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]
    topic_coherence = 0
    topic_coherence_count=0.0
    for i,word in enumerate(top_words):
      if word not in word_counts:
        count = count_word(dataset,word)
        word_counts[word]=count
      for j in range(i):
        word2 = top_words[j]
        combination = (word,word2)
        if combination not in word_combination_counts:
          count = count_word_combination(dataset,combination)
          word_combination_counts[combination]=count
        #now calculate coherence
        wc1 = word_counts[word]/float(length)
        wc2 = word_counts[word2]/float(length)
        cc = (word_combination_counts[combination])/float(length)
        if cc>0:
          coherence = tf.math.log(cc/float(wc1*wc2))/(-tf.math.log(cc))
          topic_coherence+=coherence
          coherence_sum+=coherence
        coherence_count+=1
        topic_coherence_count+=1
    topic_coherence_sum+=topic_coherence/float(topic_coherence_count)
  return coherence_sum/float(coherence_count),topic_coherence_sum/float(len(beta))
