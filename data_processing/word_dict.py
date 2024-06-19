#用于读取和写入 pickle 文件。
import pickle

#功能：从两个语料库中提取词汇集合。
#参数：corpus1: 语料库1，类型为list。orpus2: 语料库2，类型为list。
#返回：返回一个包含所有词汇的集合word_vocab。
def get_vocab(corpus1, corpus2):
    word_vocab = set()  #初始化一个空集合 word_vocab。
    for corpus in [corpus1, corpus2]:   #遍历两个语料库 corpus1 和 corpus2。
        #对每个语料库的每个元素，提取嵌套在特定索引位置的词汇，并将这些词汇添加到 word_vocab 集合中。
        for i in range(len(corpus)):
            word_vocab.update(corpus[i][1][0])
            word_vocab.update(corpus[i][1][1])
            word_vocab.update(corpus[i][2][0])
            word_vocab.update(corpus[i][3])
    print(len(word_vocab))
    return word_vocab


#功能：加载并返回pickle文件中的数据。
#参数：filename：包含pickle数据的文件名。
#返回：返回从pickle文件加载的数据 data。
def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


#功能：处理输入的两个语料文件，从中提取词汇集合并进行去重操作，然后将最终的词汇集合保存到指定文件中。
#参数：filepath1：包含词汇数据的文件路径。filepath2：包含语料库数据的文件路径。save_path：保存处理后的词汇表的文件路径。
def vocab_processing(filepath1, filepath2, save_path):
    #打开并读取 filepath1 中的词汇数据，将其转换为集合 total_data1。
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
    #打开并读取 filepath2 中的语料库数据，将其存储在 total_data2。
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())
    #使用 get_vocab 函数从 total_data2 中提取词汇集合 word_set。
    word_set = get_vocab(total_data2, total_data2)
    #计算 total_data1 和 word_set 的交集 excluded_words。
    excluded_words = total_data1.intersection(word_set)
    #从 word_set 中移除 excluded_words。
    word_set = word_set - excluded_words

    print(len(total_data1))
    print(len(word_set))

    with open(save_path, 'w') as f:
        f.write(str(word_set))

#功能：程序通过 if name == "main": 检查是否为脚本直接运行，并在直接运行时执行以下操作：1）定义文件路径。2）调用 vocab_processing 函数处理语料数据并保存结果。
if __name__ == "__main__":
    #定义多个文件路径变量，包括 Python 和 SQL 数据文件路径。
    python_hnn = './data/python_hnn_data_teacher.txt'
    python_staqc = './data/staqc/python_staqc_data.txt'
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = './data/sql_hnn_data_teacher.txt'
    sql_staqc = './data/staqc/sql_staqc_data.txt'
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'
    #调用 vocab_processing 函数，处理 SQL 数据文件，并将结果保存到指定的路径 large_word_dict_sql 中。
    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)
