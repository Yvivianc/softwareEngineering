# 软件工程实验
20211050204 袁金玲

## 目录
- [一、项目概述](#一项目概述)
- [二、结构说明](#二结构说明)
- [三、文件说明](#三文件说明)
  - [2.1 process_single_corpus.py文件](#process_single_corpuspy文件)
  - [2.2 word_dict.py文件](#word_dictpy文件)
  - [2.3 python_structured.py文件](#python_structuredpy文件)
  - [2.4 sql_structured.py文件](#sql_structuredpy文件)
  - [2.5 getSru2Vec.py文件](#getsru2vecpy文件)
  - [2.6 embddings_process.py文件](#embddings_processpy文件)


## 一、项目概述
  此项目的目的是对文本数据进行预测处理。后期我们通过在程序内增加详细的注释，提高程序的可读性、可维护性和可扩展性。


## 二、结构说明
```
├── hnn_preprocessing  
│   └── embaddings_process.py  
│   └── getStru2Vec.py
│   └── process_single_corpus.py
│   └── python_structured.py
│   └── sqlang_structured.py
│   └── word_dirt.py

```
## 三、文件说明

### process_single_corpus.py

#### 3.1.1 概述
  这个程序主要用于处理两种格式的数据文件（文本文件和 pickle 文件），将它们根据问题ID（qids）拆分成单问题和多问题两类数据，并且对单问题数据进行标签化处理。
#### 3.1.2 依赖库
 - `pickle`：用于读取和写入 pickle 文件
 - `Counter`：通过计数确定每个问题ID在数据中出现的频次，从而将数据分为单问题和多问题。

#### 3.1.3 函数解释

- `load_pickle(filename)`
  - 功能：加载pickle格式的文件。
  - 参数：文件路径filename。
  - 返回：反序列化后的数据

- `split_data(total_data, qids)`
  - 功能：根据问题ID（qids）将数据拆分成单问题和多问题的两个列表。
  - 参数：total_data：总的数据列表。qids：数据中所有问题的ID列表。
  - 返回：两个列表total_data_single（单问题数据）和total_data_multiple（多问题数据）。

- `data_staqc_processing(filepath, save_single_path, save_multiple_path)`
  - 功能：处理文本文件格式的数据，将其分为单问题和多问题后分别保存到指定路径的文件中。
  - 参数：filepath：需处理的文本文件路径。save_single_path：保存单问题数据的文件路径。save_multiple_path：保存多问题数据的文件路径。

- `data_large_processing(filepath, save_single_path, save_multiple_path)`
  - 功能：加载pickle文件中的数据，拆分成单问题和多问题，并以二进制形式保存到指定路径的文件中。
  - 参数：filepath：需处理的pickle文件路径。save_single_path：保存单问题数据的文件路径。save_multiple_path：保存多问题数据的文件路径。
  
- `single_unlabeled_to_labeled(input_path, output_path)`
  - 功能：将未标记的单问题数据转换为带有标签的数据。加载pickle文件，为每个数据项添加标签（1），然后将数据排序并保存到指定路径的文件中。
  - 参数：input_path：输入的单问题数据文件路径。output_path：输出的带有标签的数据文件路径。

- `主程序`
  - 功能：__main__部分加载不同路径的文本和pickle文件，通过调用上述函数处理数据，并将结果保存到各自的指定路径中。

---
### word_dict.py
#### 3.2.1 概述
  这个程序主要用于处理两个语料库文件，生成包含指定词汇集合的字典文件。通过将每个语料库中的词汇集合与另一个集合进行比对，从中删除相同的部分，并将最终的词汇集合保存到新的文件中。
#### 3.2.2 依赖库
 - `pickle`：用于读取和写入 pickle 文件

#### 3.2.3 函数解释

- `get_vocab(corpus1, corpus2)`
  - 功能：从两个语料库中提取词汇集合。
  - 参数：corpus1: 语料库1，类型为list。orpus2: 语料库2，类型为list。
  - 返回：返回一个包含所有词汇的集合word_vocab。

- `load_pickle(filename)`
  - 功能：加载并返回pickle文件中的数据。
  - 参数：filename：包含pickle数据的文件名。
  - 返回：返回从pickle文件加载的数据 data。

- `vocab_processing(filepath1, filepath2, save_path)`
  - 功能：处理输入的两个语料文件，从中提取词汇集合并进行去重操作，然后将最终的词汇集合保存到指定文件中。
  - 参数：filepath1：包含词汇数据的文件路径。filepath2：包含语料库数据的文件路径。save_path：保存处理后的词汇表的文件路径。

- `主程序`
  - 功能：程序通过 if __name__ == "__main__": 检查是否为脚本直接运行，并在直接运行时执行以下操作：1）定义文件路径。2）调用 vocab_processing 函数处理语料数据并保存结果。

---
### python_structured.py
#### 3.3.3 概述
  这个程序是一个用于解析代码和自然语言的工具，提供了一系列函数来处理输入的代码和文本，并将它们转换为标记化形式。具体功能包括修复代码输入/输出格式、提取代码中的变量名、将代码解析为标记、对自然语言文本进行预处理和分词、还原缩略词等。这些处理步骤有助于对代码和自然语言进行进一步的分析和处理。
#### 3.3.2 依赖库
 - `re`：用于正则表达式匹配和替换操作。
 - `ast`:用于对 Python 代码进行抽象语法树解析。
 - `sys`:用于程序与解释器交互。
 - `token`：用于定义 Python 代码中的 token 类型。
 - `tokenize`：用于解析 Python 代码中的 token。
 - `io.StringIO`：用于在内存中处理字符串的文件对象。
 - `inflection`：处理单词复数形式、驼峰命名法转换等。
 - `nltk`：自然语言处理工具包，用于词性标注、分词和词形还原。

#### 3.3.3 函数解释
- `repair_program_io(code)`
  - 功能：修复包含输入输出标记的代码片段，使其变得可执行。这个函数首先定义了一些正则表达式模式，用于匹配不同类型的输入输出标记。然后，它将代码按行分割，并使用正则表达式模式匹配每一行，以确定该行是否包含输入输出标记。根据这些匹配结果，函数会对代码进行修复，去除不必要的标记行，使代码能够正确执行。
  - 参数：code：待修复的 Python 代码字符串。
  - 返回：repaired_code：修复后的代码字符串，code_list：修复后的代码片段列表。

- `get_vars(ast_root)`
  - 功能：从抽象语法树中提取变量名。
  - 参数：ast_root：Python 抽象语法树的根节点。
  - 返回：代码中出现的变量名的有序列表。


- `get_vars_heuristics(code)`
  - 功能：通过启发式方法从代码中提取变量名。
  - 参数：code：待处理的 Python 代码字符串。
  - 返回：变量名集合。

- `PythonParser(code)`
  - 功能：解析并标记化Python代码，提取变量名并处理代码标记。
  - 参数：code：待处理的 Python 代码字符串。
  - 返回：tokenized_code：标记化的单词列表，bool_failed_var：是失败变量标志，bool_failed_token：失败标记标志。
└──`first_trial(_code)`:尝试对输入代码进行标记化，如果成功返回True，否则返回False。

- `revert_abbrev(line)`
  - 功能：使用正则表达式识别并展开常见的英文缩写。
  - 参数：line:包含缩写的输入行。
  - 返回：line:展开的文本行。


- `get_wordpos(tag)`
  - 功能：将NLTK的词性标签转换为WordNet词性标签，以便于词形还原。
  - 参数：tag:来自NLTK的词性标签
  - 返回：WordNet词性标签或None。

- `process_nl_line(line)`
  - 功能：对输入的自然语言文本行进行清理和处理，包括标记化、词性标注和词形还原。
  - 参数：line:输入的文本行。
  - 返回：ine:清理和处理后的文本行。

- `process_sent_word(line)`
  - 功能：对一行自然语言文本进行分词、词性标注和词干提取。
  - 参数：line:输入的文本行。
  - 返回：word_list:处理后的单词列表。

- `filter_all_invachar(line)`
  - 功能：使用正则表达式去除文本中的非常用符号，包括多余的横线和下划线，同时将特定符号替换为空格。
  - 参数：line:输入的文本行。
  - 返回：line:清理后的文本行。

- `filter_part_invachar(line)`
  - 功能：使用正则表达式去除文本中的非常用符号，包括多余的横线和下划线，同时将特定符号替换为空格。
  - 参数：line:输入的文本行。
  - 返回：line:清理后的文行。

- `python_code_parse(line)`
  - 功能：解析和标记化Python代码。
  - 参数：line：入的Python代码字符串。
  - 返回：token_list:标记化后的代码标记列表。如果解析失败，返回'-1000'。

- `python_query_parse(line)`
  - 功能：处理和解析自然语言查询。
  - 参数：line:输入的自然语言查询字符串。
  - 返回：word_list:处理后的单词列表。

- `python_context_parse(line)`
  - 功能：处理和解析自然语言上下文。
  - 参数：line:输入的自然语言上下文字符串。
  - 返回：word_list:处理后的单词表。

- `__name__ == '__main__'`语句块
  - 功能：通过调用之前定义的几个函数，对自然语言查询、自然语言上下文和Python代码进行解析和处理。
 
---
### sql_structured.py
#### 3.4.1. 概述
  该程序主要是用来解析 SQL 代码，修复代码中不一致的变量命名问题，并对代码进行一定程度的重构，包括生成代码注释。它还可以处理自然语言查询，生成结构化的 token 列表，便于进一步的分析和处理。
#### 3.4.2. 依赖库
 - `re`：用于正则表达式匹配和替换
 - `sqlparse`：sql解析
 - `inflection`：用于进行单词的单复数转换
 - `nltk`：自然语言处理工具包，用于词性标注、分词和词形还原

#### 3.4.3. 函数解释

- `tokenizeRegex(s)`
  - 功能：使用预定义的正则表达式模式对输入字符串进行分词，并返回分词结果。
  - 参数：s：需要进行正则表达式分词的输入字符串。
  - 返回：分词结果列表。

- `sanitizeSql(sql)`
  - 功能：用于对输入的 SQL 语句进行预处理，使其格式规范化。
  - 参数：sql:输入的SQL语句。
  - 返回：规范化后的SQL字符串。

- `parseStrings(self, tok)`
  - 功能：解析并处理SQL中的字符串令牌。
  - 参数：tok：一个 SQL 解析树的节点。


- `renameIdentifiers(self, tok)`
  - 功能：重新命名SQL中的标识符（列名和表名）。
  - 参数：tok：一个 SQL 解析树的节点。

- `__hash__(self)`
  - 功能：生成解析树的哈希值

- `__init__(self, sql, regex=False, rename=True)`
  - 功能：初始化对象。
  - 参数：sql:输入的SQL语句。regex:是否对字符串使用正则表达式进行处理。rename:是否重命名标识符。

- `getTokens(parse)`
  - 功能：遍历解析树，将所有令牌展开并加入列表。
  - 参数：parse：解析树。
  - 返回：所有令牌的列表。

- `removeWhitespaces(self, tok)`
  - 功能：递归遍历解析树，移除所有空白符号。
  - 参数：tok：一个SQL解析树的节点。

- `identifySubQueries(self, tokenList)`
  - 功能：识别解析树中的子查询。递归遍历解析树，标记并识别子查询。
  - 参数：tokenList：解析树。
  - 返回：布尔值，表示是否存在子查询。

- `identifyLiterals(self, tokenList)`
  - 功能：识别解析树中的字面量和特殊标记。递归遍历解析树，标记各类字面量和特殊标记。
  - 参数：tokenList：解析树。

- `identifyFunctions(self, tokenList)`
  - 功能：识别解析树中的函数调用。递归遍历解析树，标记函数调用。
  - 参数：tokenList：解析树。

- `identifyTables(self, tokenList)`
  - 功能：识别解析树中的表引用。递归遍历解析树，标记表引用。
  - 参数：tokenList：解析树。

- `__str__(self)`
  - 功能：将解析树中的所有令牌连接成一个字符串。
  - 返回：解析树的字符串表示。

- `parseSql(self)`
  - 功能：返回解析后的令牌列表。
  - 返回：解析后的令牌列表。

- `revert_abbrev(line)`
  - 功能：处理缩略词，将其还原成完整形式。
  - 参数：line:输入的带有缩略词的句子。
  - 返回：还原缩写后的句子。

- `get_wordpos(tag)`
  - 功能：将 POS 标签转换为 WordNet 词性标签。
  - 参数：tag: POS 标签。
  - 返回：WordNet 词性标签或 None

- `process_nl_line(line)`
  - 功能：预处理自然语言句子，包括去除冗余字符和骆驼命名法转换。
  - 参数：line:需要处理的句子。
  - 返回：处理后的字符串。

- `process_sent_word(line)`
  - 功能：对句子进行分词和处理，包括词性标注、词形还原和词干提取。
  - 参数：line:需要处理的句子。
  - 返回：处理后的词列表。

- `filter_all_invachar(line)`
  - 功能：去除字符串中的非常用符号，防止解析出错。
  - 参数：line:需要处理的文本行。
  - 返回：处理后的字符串。

- `filter_part_invachar(line)`
  - 功能：去除字符串中的部分非常用符号，防止解析出错。
  - 参数：line:需要处理的文本行。
  - 返回：处理后的字符串。

- `sqlang_code_parse(line)`
  - 功能：解析SQL代码，生成标记(token)列表。
  - 参数：line:输入的SQL代码行。
  - 返回：解析后的标记列表或错误代码。

- `sqlang_query_parse(line)`
  - 功能：解析自然语言查询，生成标记(token)列表。
  - 参数：line:需要处理的查询行。
  - 返回：解析后的标记列表。

- `sqlang_context_parse(line)`
  - 功能：解析上下文信息，生成标记(token)列表。
  - 参数：line:需要处理的上下文行。
  - 返回：解析后的标记列表。

- `__name__ == '__main__'`语句块
  - 功能：测试前面定义的函数 sqlang_code_parse 和 sqlang_query_parse 的工作是否正常。

---
### getStru2Vec.py
#### 3.5.1. 概述
  这个程序的主要功能是利用多进程并行处理Python和SQL的未标注数据。这些数据通过解析函数生成上下文（context）、查询（query）和代码（code）数据块，然后将其存储在文件中。程序中定义的各个函数实现了数据的解析、并行处理和数据的保存。

#### 3.5.2. 依赖库
 - `pickle`：用于读取和写入 pickle 文件。
 - `multiprocessing`:用于创建并行子进程来加速数据处理。
 - `python_structured`和`sqlang_structured`：自定义模块，包含处理Python和SQL语言代码的函数。

#### 3.5.3. 函数解释
- `multipro_python_query(data_list)`
  - 功能：并行处理Python代码查询数据。
  - 参数：data_list：输入的数据列表, 包含所有待处理的Python查询数据。
  - 返回：解析后的Python查询列表。

- `multipro_python_code(data_list)`
  - 功能：并行处理Python代码数据。
  - 参数：data_list：输入的数据列表, 包含所有待处理的Python代码数据。
  - 返回：解析后的Python代码列表。

- `multipro_python_context(data_list)`
  - 功能：并行处理Python代码上下文数据。
  - 参数：data_list：输入的数据列表, 包含所有待处理的Python上下文数据。
  - 返回：解析后的Python上下文列表。如果输入的数据等于-10000，则返回['-10000']。

- `multipro_sqlang_query(data_list)`
  - 功能：并行处理SQL代码查询数据。
  - 参数：data_list：输入的数据列表, 包含所有待处理的SQL查询数据。
  - 返回：解析后的SQL查询列表。

- `multipro_sqlang_code(data_list)`
  - 功能：并行处理SQL代码数据。
  - 参数：data_list：输入的数据列表, 包含所有待处理的SQL代码数据。
  - 返回：解析后的代码列表。

- `multipro_sqlang_context(data_list)`
  - 功能：并行处理SQL代码上下文数据。
  - 参数：data_list：输入的数据列表，包含所有待处理的SQL上下文数据。
  - 返回：解析后的SQL上下文列表。如果输入的数据等于-10000，则返回['-10000']。

- `parse(data_list, split_num, context_func, query_func, code_func)`
  - 功能：并行处理输入数据，提取上下文、查询和代码信息。
  - 参数：data_list：输入的数据列表。split_num：数据拆分的块大小。context_func：处理上下文的函数。query_func：处理查询的函数。code_func：处理代码的函数。
  - 返回：context_data, query_data, code_data - 处理后的上下文、查询和代码数据列表。

- `main(lang_type,split_num,source_path,save_path,context_func,query_func,code_func)`
  - 功能：加载数据、调用 parse 函数处理数据，并将结果保存为二进制文件。
  - 参数：lang_type：语言类型。split_num：数据拆分的块大小。source_path：输入数据的路径。save_path：保存处理后数据的路径。context_func：处理上下文的函数。query_func：处理查询的函数。code_func：处理代码的函数。

---
### embaddings_process.py
#### 3.6.1. 概述
  这个程序主要用于处理词向量和文本数据的序列化。它包括将文本格式的词向量转换为二进制格式，构建新的词典和词向量矩阵，以及将文本数据序列化以便后续处理。
#### 3.6.2. 依赖库
 - `pickle`：用于读取和写入 pickle 文件。
 - `numpy`：用于数值计算和数组处理。
 - `gensim.models.KeyedVectors`：用于加载和保存词向量模型

#### 3.6.3. 类和方法说明
- `trans_bin(path1, path2)`
  - 功能：将文本格式的词向量文件转换为二进制格式的文件，并保存在指定路径。
  - 参数：path1：待转换的文本格式的词向量文件路径。path2：转换后的二进制格式的词向量文件路径。

- `get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path)`
  - 功能：构建新的词典和词向量矩阵。
  - 参数：type_vec_path：词向量文件路径。type_word_path：词典文件路径。final_vec_path：输出的词向量文件路径。final_word_path：输出的词典文件路径。

- `get_index(type, text, word_dict)`
  - 功能：获取文本在词典中的位置索引。
  - 参数：type：类型（‘code’ 或 ‘text’）。text：待处理的文本。word_dict：词典。
  - 返回：位置索引列表


- `serialization(word_dict_path, type_path, final_type_path)`
  - 功能：将训练、测试或验证语料序列化。
  - 参数：word_dict_path：词典文件路径。type_path：包含语料的文件路径。final_type_path：序列化后的文件保存路径。

- `__name__ == '__main__'`语句块
  - 功能：定义各种文件路径，包括词向量文件、词典文件和待处理的语料文件路径。
    调用 serialization 函数处理Python语言的语料，并将结果保存到指定路径。
---


