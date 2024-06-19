#用于读取和写入 pickle 文件。
import pickle
#用于创建并行子进程来加速数据处理。
import multiprocessing
#自定义模块，包含处理Python和SQL语言代码的函数。
from python_structured import *
from sqlang_structured import *


#功能：并行处理Python代码查询数据。
#参数：data_list：输入的数据列表, 包含所有待处理的Python查询数据。
#返回：解析后的Python查询列表。
def multipro_python_query(data_list):
    return [python_query_parse(line) for line in data_list]


#功能：并行处理Python代码数据。
#参数：data_list：输入的数据列表, 包含所有待处理的Python代码数据。
#返回：解析后的Python代码列表。
def multipro_python_code(data_list):
    return [python_code_parse(line) for line in data_list]


#功能：并行处理Python代码上下文数据。
#参数：data_list：输入的数据列表, 包含所有待处理的Python上下文数据。
#返回：解析后的Python上下文列表。如果输入的数据等于-10000，则返回['-10000']。
def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result


#功能：并行处理SQL代码查询数据。
#参数：data_list：输入的数据列表, 包含所有待处理的SQL查询数据。
#返回：解析后的SQL查询列表。
def multipro_sqlang_query(data_list):
    return [sqlang_query_parse(line) for line in data_list]


#功能：并行处理SQL代码数据。
#参数：data_list：输入的数据列表, 包含所有待处理的SQL代码数据。
#返回：解析后的代码列表。
def multipro_sqlang_code(data_list):
    return [sqlang_code_parse(line) for line in data_list]


#功能：并行处理SQL代码上下文数据。
#参数：data_list：输入的数据列表，包含所有待处理的SQL上下文数据。
#返回：解析后的SQL上下文列表。如果输入的数据等于-10000，则返回['-10000']。
def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result


#功能：并行处理输入数据，提取上下文、查询和代码信息。
#参数：data_list：输入的数据列表。split_num：数据拆分的块大小。context_func：处理上下文的函数。query_func：处理查询的函数。code_func：处理代码的函数。
#返回：context_data, query_data, code_data - 处理后的上下文、查询和代码数据列表。
def parse(data_list, split_num, context_func, query_func, code_func):
    #初始化进程池
    pool = multiprocessing.Pool()
    #数据拆分
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    #处理上下文数据
    results = pool.map(context_func, split_list)
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')
    #处理查询数据
    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')
    #处理代码数据
    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')
    #关闭进程池并等待所有进程完成
    pool.close()
    pool.join()

    return context_data, query_data, code_data


#功能：加载数据、调用 parse 函数处理数据，并将结果保存为二进制文件。
#参数：lang_type：语言类型。split_num：数据拆分的块大小。source_path：输入数据的路径。
#参数：save_path：保存处理后数据的路径。context_func：处理上下文的函数。query_func：处理查询的函数。code_func：处理代码的函数。
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)
    #并行处理数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    #提取标识符
    qids = [item[0] for item in corpus_lis]
    #组装总数据
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]
    #保存处理后的数据
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

if __name__ == '__main__':
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)
