#用于读取和写入 pickle 文件。
import pickle 
#用于数值计算和数组处理。
import numpy as np  
#用于加载和保存词向量模型
from gensim.models import KeyedVectors 


#将文本格式的词向量文件转换为二进制格式的文件，并保存在指定路径。
#将词向量文件保存为二进制文件
#trans_bin 函数接受两个参数：path1 是待转换的文本格式的词向量文件路径，path2 是转换后的二进制格式的词向量文件路径。
def trans_bin(path1, path2):
    #使用 KeyedVectors.load_word2vec_format 方法加载文本格式的词向量文件，该文件包含了预训练的词向量。参数 binary=False 表示加载的是文本格式的文件。
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    #在加载完词向量后，调用 init_sims 方法对词向量进行归一化处理，这可以提高查询速度，并节省内存空间。
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)


#功能：构建新的词典和词向量矩阵。
#参数：type_vec_path：词向量文件路径。type_word_path：词典文件路径。final_vec_path：输出的词向量文件路径。final_word_path：输出的词典文件路径。
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')
    # 读取词典文件内容
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())

    # 输出词向量
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 其中0 PAD_ID, 1 SOS_ID, 2 EOS_ID, 3 UNK_ID

    fail_word = []  # 用于记录无法找到向量的词汇
    rng = np.random.RandomState(None)  # 随机数生成器，用于生成UNK等特殊词的随机向量
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze() # UNK的向量为随机值
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze() # SOS的向量为随机值
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze() # EOS的向量为随机值
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]
    
    # 遍历总词汇列表
    for word in total_word:
        try:
            # 尝试从模型中获取词向量
            word_vectors.append(model.wv[word])  # 加载词向量
            word_dict.append(word)
        except:
            # 如果无法获取词向量，记录该词
            fail_word.append(word)

    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    
    # 将词向量保存到指定路径
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    # 将词典保存到指定路径
    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")



#获取文本在词典中的位置索引。
#参数：type：类型（‘code’ 或 ‘text’）。text：待处理的文本。word_dict：词典。
#返回：位置索引列表
def get_index(type, text, word_dict):
    location = []
    if type == 'code':
        location.append(1)  # 1表示起始符
        len_c = len(text)
        if len_c + 1 < 350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)  # 2表示结束符
            else:
                for i in range(0, len_c):
                    # 获取词的索引，若词不在词典中则使用UNK的索引
                    index = word_dict.get(text[i], word_dict['UNK'])
                    location.append(index)
                location.append(2)
        else:
            for i in range(0, 348):
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)
            location.append(2)
    else: # 处理普通文本
        if len(text) == 0:   # 若文本为空，则返回0
            location.append(0)  
        elif text[0] == '-10000': # 特殊标记处理
            location.append(0)
        else:
            for i in range(0, len(text)):
                # 获取词的索引，若词不在词典中则使用UNK的索引
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)

    return location


# 将训练、测试、验证语料序列化
# 查询：25 上下文：100 代码：350
def serialization(word_dict_path, type_path, final_type_path):
    # 加载词典
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)
    # 加载语料库
    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for i in range(len(corpus)):
        qid = corpus[i][0]
        # 将文本和代码转换为索引表示
        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)
        tokenized_code = get_index('code', corpus[i][2][0], word_dict)
        query_word_list = get_index('text', corpus[i][3], word_dict)
        block_length = 4
        label = 0
        # 对文本和代码进行截断或填充
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (100 - len(Si1_word_list))
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (25 - len(query_word_list))
        # 构建一个数据实例
        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        # 添加到总数据列表中
        total_data.append(one_data)
    # 序列化并保存处理后的数据
    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)

#定义各种文件路径，包括词向量文件、词典文件和待处理的语料文件路径。
if __name__ == '__main__':
    # 词向量文件路径
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # ==========================最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================

    # sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sqlfinal_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)

    # python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    # python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)

    # 处理成打标签的形式
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    # test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)
