#用于读取和写入 pickle 文件
import pickle
#通过计数确定每个问题ID在数据中出现的频次，从而将数据分为单问题和多问题。
from collections import Counter


#功能：加载pickle格式的文件。
#参数：文件路径filename。
#返回：反序列化后的数据
def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data


#功能根据问题ID（qids）将数据拆分成单问题和多问题的两个列表。
#参数：total_data：总的数据列表。qids：数据中所有问题的ID列表。
#返回：两个列表total_data_single（单问题数据）和total_data_multiple（多问题数据）。
def split_data(total_data, qids):
    #计数 qids 中每个标识符的出现次数
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    #遍历 total_data 并分类数据项
    for data in total_data:
        if result[data[0][0]] == 1:
            total_data_single.append(data)
        else:
            total_data_multiple.append(data)
    return total_data_single, total_data_multiple


#功能：处理文本文件格式的数据，将其分为单问题和多问题后分别保存到指定路径的文件中。
#参数：filepath：需处理的文本文件路径。save_single_path：保存单问题数据的文件路径。save_multiple_path：保存多问题数据的文件路径。
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    with open(filepath, 'r') as f:
        total_data = eval(f.read())  #从给定路径的文件中读取数据
    #将数据分成单问题和多问题两个列表，并将结果写入到指定路径的文件中。
    qids = [data[0][0] for data in total_data]
    total_data_single, total_data_multiple = split_data(total_data, qids)

    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


#功能：加载pickle文件中的数据，拆分成单问题和多问题，并以二进制形式保存到指定路径的文件中。
#参数：filepath：需处理的pickle文件路径。save_single_path：保存单问题数据的文件路径。save_multiple_path：保存多问题数据的文件路径。
def data_large_processing(filepath, save_single_path, save_multiple_path):
    total_data = load_pickle(filepath)
    #将数据分成单问题和多问题两个列表，并将结果以二进制形式写入到指定路径的文件中。
    qids = [data[0][0] for data in total_data]
    total_data_single, total_data_multiple = split_data(total_data, qids)

    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


#功能：将未标记的单问题数据转换为带有标签的数据。加载pickle文件，为每个数据项添加标签（1），然后将数据排序并保存到指定路径的文件中。
#参数：input_path：输入的单问题数据文件路径。output_path：输出的带有标签的数据文件路径。
def single_unlabeled_to_labeled(input_path, output_path):
    total_data = load_pickle(input_path)
    #加载 pickle 文件，然后为每个数据项添加标签（1）
    labels = [[data[0], 1] for data in total_data]
    #根据问题ID和标签进行排序，最后将结果写入到指定路径的文件中。
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))


#功能：__main__部分加载不同路径的文本和pickle文件，通过调用上述函数处理数据，并将结果保存到各自的指定路径中。
if __name__ == "__main__":
    #读取未标记的问题数据文件 staqc_python_path，然后将单问题和多问题数据分别保存到指定路径的文件中。
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)
