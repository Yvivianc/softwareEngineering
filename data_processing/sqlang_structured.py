# -*- coding: utf-8 -*-
import re            #用于正则表达式匹配和替换
import sqlparse #0.4.2 #sql解析

#骆驼命名法
import inflection #用于进行单词的单复数转换和骆驼命名法
 
#词性还原
#nltk：处理单词复数形式、驼峰命名法转换等。
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

#词干提取
from nltk.corpus import wordnet

#############################################################################
#常量定义
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4

TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11

SUBQUERY = 12

DUD = 13
#ttypes 字典
ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
          8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD", }
#正则表达式扫描器 scanner
scanner = re.Scanner([(r"\[[^\]]*\]", lambda scanner, token: token), (r"\+", lambda scanner, token: "REGPLU"),
                      (r"\*", lambda scanner, token: "REGAST"), (r"%", lambda scanner, token: "REGCOL"),
                      (r"\^", lambda scanner, token: "REGSTA"), (r"\$", lambda scanner, token: "REGEND"),
                      (r"\?", lambda scanner, token: "REGQUE"),
                      (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
                      (r'.', lambda scanner, token: None), ])

#---------------------子函数1：代码的规则--------------------
#功能：使用预定义的正则表达式模式对输入字符串进行分词，并返回分词结果。
def tokenizeRegex(s):
    results = scanner.scan(s)[0]
    return results

#---------------------子函数2：代码的规则--------------------
class SqlangParser():
    @staticmethod
    #功能：用于对输入的 SQL 语句进行预处理，使其格式规范化。
    #参数：sql:输入的SQL语句。
    #返回：规范化后的SQL字符串。
    def sanitizeSql(sql):
        #去除空白和小写转换
        s = sql.strip().lower()
        #确保SQL语句以分号结尾
        if not s[-1] == ";":
            s += ';'
        #括号两侧添加空格
        s = re.sub(r'\(', r' ( ', s)
        s = re.sub(r'\)', r' ) ', s)
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        #关键字处理
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)
        #去除井号
        s = s.replace('#', '')
        return s

    #功能：解析并处理SQL中的字符串令牌。
    #参数：tok：一个 SQL 解析树的节点。
    def parseStrings(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            #递归处理子令牌
            for c in tok.tokens:
                self.parseStrings(c)
        elif tok.ttype == STRING:  #处理字符串令牌
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))
            else:
                tok.value = "CODSTR"
    
    #功能：重新命名SQL中的标识符（列名和表名）。
    #参数：tok：一个 SQL 解析树的节点。
    def renameIdentifiers(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)
        elif tok.ttype == COLUMN:  #列名重命名
            if str(tok) not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])
                self.idMap["COLUMN"][str(tok)] = colname
                self.idMapInv[colname] = str(tok)
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][str(tok)]  #将 tok.value 更新为新名称。
        elif tok.ttype == TABLE:  #表名重命名
            if str(tok) not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])
                self.idMap["TABLE"][str(tok)] = tabname
                self.idMapInv[tabname] = str(tok)
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][str(tok)]

        elif tok.ttype == FLOAT:  #以下均为数值类型重命名
            tok.value = "CODFLO"
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"
        elif tok.ttype == HEX:
            tok.value = "CODHEX"

    #功能：生成解析树的哈希值
    def __hash__(self):
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))
    
    #功能：初始化对象。
    def __init__(self, sql, regex=False, rename=True):
        #属性初始化
        self.sql = SqlangParser.sanitizeSql(sql)
        
        self.idMap = {"COLUMN": {}, "TABLE": {}}
        self.idMapInv = {}
        self.idCount = {"COLUMN": 0, "TABLE": 0}
        self.regex = regex

        self.parseTreeSentinel = False
        self.tableStack = []
        #SQL解析和预处理
        self.parse = sqlparse.parse(self.sql)
        self.parse = [self.parse[0]]
        #去除空白字符和识别字面量
        self.removeWhitespaces(self.parse[0])
        self.identifyLiterals(self.parse[0])
        #子查询和函数识别
        self.parse[0].ptype = SUBQUERY
        self.identifySubQueries(self.parse[0])
        self.identifyFunctions(self.parse[0])
        self.identifyTables(self.parse[0])
        #字符串解析和标识符重命名
        self.parseStrings(self.parse[0])

        if rename:
            self.renameIdentifiers(self.parse[0])
        #获取Token
        self.tokens = SqlangParser.getTokens(self.parse)

    @staticmethod
    #功能：遍历解析树，将所有令牌展开并加入列表。
    #参数：parse：解析树。
    #返回：所有令牌的列表。
    def getTokens(parse):
        flatParse = []
        for expr in parse:
            #扁平化处理
            for token in expr.flatten():  #处理字符串标记
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse
    
    #功能：递归遍历解析树，移除所有空白符号。
    #参数：tok：一个SQL解析树的节点。
    def removeWhitespaces(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []
            #遍历并筛选非空白标记
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmpChildren.append(c)

            tok.tokens = tmpChildren
            #递归调用移除空白字符
            for c in tok.tokens:
                self.removeWhitespaces(c)

    #功能：识别解析树中的子查询。递归遍历解析树，标记并识别子查询。
    #参数：tokenList：解析树。
    #返回：布尔值，表示是否存在子查询。
    def identifySubQueries(self, tokenList):
        isSubQuery = False

        for tok in tokenList.tokens:
            #递归处理TokenList
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)
                #检查并标记子查询
                if (subQuery and isinstance(tok, sqlparse.sql.Parenthesis)):
                    tok.ttype = SUBQUERY
                #检查select关键字
            elif str(tok) == "select":
                isSubQuery = True
        return isSubQuery
    
    #功能：识别解析树中的字面量和特殊标记。递归遍历解析树，标记各类字面量和特殊标记。
    #参数：tokenList：解析树。
    def identifyLiterals(self, tokenList):
        #初始化标记类型
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blankTokenTypes = [sqlparse.sql.Identifier]

        for tok in tokenList.tokens:
            #处理TokenList类型的标记
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identifyLiterals(tok)
            #处理关键字
            elif (tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select"):
                tok.ttype = KEYWORD
            #处理整数
            elif (tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer):
                tok.ttype = INTEGER
            #处理十六进制数
            elif (tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal):
                tok.ttype = HEX
            #处理浮点数
            elif (tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float):
                tok.ttype = FLOAT
            #处理字符串
            elif (tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol):
                tok.ttype = STRING
            #处理通配符
            elif (tok.ttype == sqlparse.tokens.Wildcard):
                tok.ttype = WILDCARD
            #处理列标识符
            elif (tok.ttype in blankTokens or isinstance(tok, blankTokenTypes[0])):
                tok.ttype = COLUMN
    
    #功能：识别解析树中的函数调用。递归遍历解析树，标记函数调用。
    #参数：tokenList：解析树。
    def identifyFunctions(self, tokenList):
        for tok in tokenList.tokens:
            #识别函数和括号
            if (isinstance(tok, sqlparse.sql.Function)):
                self.parseTreeSentinel = True
            elif (isinstance(tok, sqlparse.sql.Parenthesis)):
                self.parseTreeSentinel = False
            #标记函数类型
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            #递归处理TokenList
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)
    
    #功能：识别解析树中的表引用。递归遍历解析树，标记表引用。
    #参数：tokenList：解析树。
    def identifyTables(self, tokenList):
        #处理子查询
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)
        #遍历标记列表
        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]
            tok = tokenList.tokens[i]
            #识别表标记和关键字
            if (str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN):
                prevtok.ttype = TABLE

            elif (str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = True

            elif ((str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = False
            #递归处理TokenList
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)
            #识别列标记
            elif (tok.ttype == COLUMN):
                if self.tableStack[-1]:
                    tok.ttype = TABLE
        #处理子查询结束
        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()
    
    #功能：将解析树中的所有令牌连接成一个字符串。
    def __str__(self):
        return ' '.join([str(tok) for tok in self.tokens])

    #功能：返回解析后的令牌列表。
    def parseSql(self):
        return [str(tok) for tok in self.tokens]
#############################################################################

#############################################################################
#缩略词处理
#功能：处理缩略词，将其还原成完整形式。
#参数：line:输入的带有缩略词的句子。
#返回：还原缩写后的句子。
def revert_abbrev(line):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")
    #替换缩写
    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line


#功能：将 POS 标签转换为 WordNet 词性标签。
#参数：tag: POS 标签。
#返回：WordNet 词性标签或 None
def get_wordpos(tag):  #获取词性
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#---------------------子函数1：句子的去冗--------------------
#功能：预处理自然语言句子，包括去除冗余字符和骆驼命名法转换。
#参数：line:需要处理的句子。
#返回：处理后的字符串。
def process_nl_line(line):
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = line.replace('\t', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^\(|^\)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除末尾.和空格
    line=line.strip()
    return line


#---------------------子函数1：句子的分词--------------------
#功能：对句子进行分词和处理，包括词性标注、词形还原和词干提取。
#参数：line:需要处理的句子。
#返回：处理后的词列表。
def process_sent_word(line):
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字 56
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符 6c60b8e1
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words= line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    #词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list=[]
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取(效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list


#############################################################################
#功能：去除字符串中的非常用符号，防止解析出错。
#参数：line:需要处理的文本行。
#返回：处理后的字符串。
def filter_all_invachar(line):
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


#功能：去除字符串中的部分非常用符号，防止解析出错。
#参数：line:需要处理的文本行。
#返回：处理后的字符串。
def filter_part_invachar(line):
    #去除非常用符号；防止解析有误
    line= re.sub('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    #包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


########################主函数：代码的tokens#################################
#功能：解析SQL代码，生成标记(token)列表。
#参数：line:输入的SQL代码行。
#返回：解析后的标记列表或错误代码。
def sqlang_code_parse(line):
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub(' +', ' ', line)

    line = re.sub('>>+', '', line)#新增加
    line = re.sub(r"\d+(\.\d+)+",'number',line)#新增加 替换小数

    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        query = SqlangParser(line, regex=True)
        typedCode = query.parseSql()
        typedCode = typedCode[:-1]
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower()  for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        # 返回列表
        return token_list
    # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'
########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################
#功能：解析自然语言查询，生成标记(token)列表。
#参数：line:需要处理的查询行。
#返回：解析后的标记列表。
def sqlang_query_parse(line):
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list

#功能：解析上下文信息，生成标记(token)列表。
#参数：line:需要处理的上下文行。
#返回：解析后的标记列表。
def sqlang_context_parse(line):
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list

#######################主函数：句子的tokens##################################

#功能：测试前面定义的函数 sqlang_code_parse 和 sqlang_query_parse 的工作是否正常。
if __name__ == '__main__':
    print(sqlang_code_parse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    print(sqlang_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    print(sqlang_code_parse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    print(sqlang_code_parse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    print(sqlang_code_parse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    print(sqlang_code_parse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    print(sqlang_code_parse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))



