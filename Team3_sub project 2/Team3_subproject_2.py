'''
Team 3

107065501 蘇玫如
107065527 段凱文

'''

from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.parse import CoreNLPParser
from nltk.parse import corenlp
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
from nltk import Tree
from nltk.parse.stanford import StanfordDependencyParser
import jieba
from collections import Counter
import os

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-11.0.2.jdk/Contents/Home/bin/java" 
os.environ["CLASSPATH"] = "/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars"
os.environ["STANFORD_MODELS"] = "/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/models"

### Input reading
f = open("小紅帽.txt", "r")

#==========抽取元素==========
## NN_Default
NN_except=['個','誰','時','兩','一','房子']
vocal = ['砰','咚']

loc = ['稻草屋','水潭','城堡','木屋子','磚屋','森林','高塔','河','奶奶家','王宮','花園','廚房','家','井里','山', '教堂', '客棧', '田野', '草原',
       '房子']

char_except = ['好孩子','身邊說','人們心中', '美好願望', '好幾', '個個', '小女兒', '美如天仙','小公主','嚴冬時節', '大雪片', '窗子邊', '針線活兒', '寒風捲著', '捲著雪片', '烏木窗臺', '針口流了', '三點血滴', '飄進窗子', '鮮紅血滴', '烏木窗臺', '血一樣', '這窗子','他們','你們','我們','弟弟','兄妹','女水妖','斧子']
family_member = ['祖父','祖母','爺爺','奶奶','外公','外婆','爸爸','媽媽','叔叔','舅舅','阿姨','嬸嬸','後母','姊姊','姐姐','繼母','姊妹','大哥',
                 '二哥','小弟', '兒子', '女兒', '夫婦', '母親']
passerby = ['公主','王子','女巫','狐狸','國王','皇后','獵人','王后','小矮人','魔鏡','野狼','妖精','精靈','僕人','灰姑娘','大臣','水妖','男孩',
            '老人', '裁縫', '上帝', '學徒', '師傅', '老太婆', '僕人', '寡婦', '小孩']

Time = ['清晨','白天','黃昏','晚上']

## Character
### 1. 文章內所有角色
temp = []
char=[]
w = []
t = []


for x in f:
    
    seg_list = jieba.cut(x)
    tokens = "/".join(seg_list)
    tokenizer = tokens.split("/")
    #print(tokenizer)

    zh_tagger = StanfordPOSTagger('chinese-nodistsim.tagger')
    #print(zh_tagger.tag(tokens.split("/")))
    for _, word_and_tag in  zh_tagger.tag(tokens.split("/")):
        word, tag = word_and_tag.split('#')
        #print(word,tag)
        w.append(word)
        t.append(tag)
        
for i in range(1,len(w)):
    if(t[i-1]=="NN" and t[i]=="NN"):
        a = w[i-1]+w[i]
        if(a not in char):
            temp.append(a)
    
    elif(t[i-1]=="JJ" and t[i]=="NN"):
        a = w[i-1]+w[i]
        if(a not in char):
            temp.append(a)
    if(t[i]=="NN" or t[i]=="NR"):
        a = w[i]
        if(a not in char):
            temp.append(a)

i = 0
while i < len(temp):
    if temp[i] in char_except:
        del temp[i]
    else:
        i += 1
            
#print(temp)

### 2. 找出親人關係
for i in range(len(temp)):
    if(temp[i] in family_member and temp[i] not in char):
        char.append(temp[i])
#print(char)

### 3. 找出第三方角色
for i in range(len(temp)):
    if(temp[i] in passerby and temp[i] not in char):
        char.append(temp[i])
#print(char)

### 4. 計算重要角色 Top3
from collections import Counter
word_counts = Counter(temp)
top_three = word_counts.most_common(3)
#print(top_three)

### 5. 增加可能重要或未偵測出的角色
for i in range(len(top_three)):
    if(top_three[i][0] not in loc and top_three[i][0] not in family_member and top_three[i][0] not in char and top_three[i][0] not in NN_except):
        char.insert(0+i, top_three[i][0])
#print(char)

### 6. 移除重複
i = 0
while i < len(char):
    if char[i] in char_except:
        del char[i]
    else:
        i += 1

### character output
print("Character: ")
print(char)

## Location
### 1. 從名詞（temp）去查找對照的 loc_default，並存在 loc_temp
from collections import Counter

loc_temp = []

for i in range(len(temp)):
    if(temp[i] in loc):
        loc_temp.append(temp[i])

word_counts = Counter(loc_temp)
        
#print(loc_temp)

### 2. 從 loc_temp 整理
word_counts = Counter(loc_temp)
loc_counts = word_counts.most_common(len(loc_temp))
#print(loc_counts)

loc_set = []

for i in range(len(loc_counts)):
    loc_set.append(loc_counts[i][0])

### Location output
print("Location: ")
print(loc_set)


#==========判斷場景切割法==========
### Input reading
f = open("小紅帽.txt", "r")
outside = 0
inside = 0

for x in f:
    if("「" in x):
        for i in range(len(loc_set)):
            if(loc_set[i] in x):
                inside+=1
    else:
        for i in range(len(loc_set)):
            if(loc_set[i] in x):
                outside+=1

#print((inside,outside))

if(inside>=outside): # "「" in x
    inside = True
else: # "「" not in x
    inside = False

#==========生成劇本==========
# Latest Version

### Input reading
f = open("小紅帽.txt", "r")

last_loc = ""
last_time=""
start=True
detect_print=True

for x in f:
    
    # Time
    # Night
    if('天黑' in x or '晚上' in x):
        time = Time[3]
    
    #evening    
    elif('黃昏' in x or '夕陽' in x):
        time = Time[2]
        
    #morning
    elif('晨' in x):
        time = Time[0]
        
    # default: 白天
    else:
        time = Time[1]
        
    #Location
    if(inside==True):
        if('「'  in x):
            for i in range(len(loc_set)): 
            # 地點 case1: 一開始
                if(loc_set[i] in x and loc_set[i]!=last_loc and start==True):
                    print("================================")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    last_loc = loc_set[i]
                    last_time = time
                    start=False
                    break
                
                if(start==True):
                    print("================================")
                    last_loc = '家'
                    print("Scene: 時/ "+time+"     景/ "+last_loc)
                    start = False
                    break
            
            # 地點 case2: 家以外的場景
                if(loc_set[i] in x and loc_set[i]!=last_loc and loc_set[i]!="家"):
                    print("================================")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    last_loc = loc_set[i]
                    last_time = time
                    break
                
                '''if(time!=last_time):
                    print("================================")
                    print("Scene: 時/ "+time+"     景/ "+last_loc)
                    last_time = time'''
    
    else: 
        #outside
        if('「'  not in x):
            for i in range(len(loc_set)): 
            # 地點 case1: 一開始
                if(loc_set[i] in x and loc_set[i]!=last_loc and start==True):
                    print("================================")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    last_loc = loc_set[i]
                    last_time = time
                    start=False
                    break
                
                if(start==True):
                    print("================================")
                    last_loc = '家'
                    print("Scene: 時/ "+time+"     景/ "+last_loc)
                    start = False
                    break
            
            # 地點 case2: 家以外的場景
                if(loc_set[i] in x and loc_set[i]!=last_loc and loc_set[i]!="家"):
                    print("================================")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    last_loc = loc_set[i]
                    last_time = time
                    break
                
                '''if(time!=last_time):
                    print("================================")
                    print("Scene: 時/ "+time+"     景/ "+last_loc)
                    last_time = time'''
            
    
        # 強迫換景
        if('隔天' in x or '第二天' in x):
            print("================================")
            print("Scene: 時/ "+time+"     景/ "+last_loc)
            last_time = time

       
    if("「" in x):
        # case 3-2
        if(x[0]=="「"):
            upper = (x.split("「"))
            two = upper[1]

            if(len(upper)>2):
                dia1 = upper[1].split("」")[0]
                sentence1 =  two.split("」")[1]
                sentence2 = sentence1
                dia2 = upper[2].split("」")[0]
                
                sentence = []
                sentence.append(sentence1)
                sentence.append(sentence2)

                dialogue = []
                dialogue.append(dia1)
                dialogue.append(dia2)

                for i in range(2):

                    seg_list = jieba.cut(sentence[i]) 
                    tokens = "/".join(seg_list)
                    tokenizer = tokens.split("/")
                    #print(tokenizer)
                    for j in range(len(char)):
                        if(char[j] in tokenizer):
                            A = char[j]
                            break
                    B=A

                    zh_dependency_parser = StanfordDependencyParser("/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser-3.9.2-models.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/models/chineseFactored.ser.gz")
                    ans = list(zh_dependency_parser.parse(tokens.split("/")))
                    for row in ans[0].triples():
                        if(row[1]=="nsubj" and row[2][0] in char):
                            A = row[2][0]
                            #print("from: "+A)
                        elif(row[1]=="compound:nn" and row[2][0] in char_except):
                            B = row[2][0]
                            #print("to: "+B)    
                for k in range(len(vocal)):
                    if(vocal[i] in dialogue[i]):
                        detect_print = False
                        break
                
                if(detect_print):
                    print(A+": "+dialogue[i])
                    print(sentence[i])
                else:
                    detect_print = True
                
                
            # case 3-1
            else:
                dialogue = upper[1].split("」")[0]
                sentence = upper[1].split("」")[1]
                
                seg_list = jieba.cut(sentence) 
                tokens = "/".join(seg_list)
                tokenizer = tokens.split("/")
                #print(tokenizer)
                for j in range(len(char)):
                        if(char[j] in tokenizer):
                            A = char[j]
                            break
                B=A
                
                zh_dependency_parser = StanfordDependencyParser("/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser-3.9.2-models.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/models/chineseFactored.ser.gz")
                ans = list(zh_dependency_parser.parse(tokens.split("/")))
                
                for row in ans[0].triples():
                    if(row[1]=="nsubj" and row[2][0] in char):
                        A = row[2][0]
                        #print("from: "+A)
                    elif(row[1]=="compound:nn" and row[2][0] in char_except):
                        B = row[2][0]
                        #print("to: "+B)

                for k in range(len(vocal)):
                    if(vocal[k] in dialogue):
                        detect_print = False
                        break
                
                if(detect_print):
                        print(A+": "+dialogue)
                        #print(sentence)
                else:
                    detect_print = True
                   
        # case 1, 2
        else:
            upper = (x.split("「"))
            two = upper[1]

            # case 2
            if(len(upper)>2):
                sentence1 = upper[0]
                dia1 =  upper[1].split("」")[0]
                sentence2 = (two.split("」"))[1]
                dia2 = upper[2].split("」")[0]
                
                sentence = []
                sentence.append(sentence1)
                sentence.append(sentence2)

                dialogue = []
                dialogue.append(dia1)
                dialogue.append(dia2)

                for i in range(2):

                    seg_list = jieba.cut(sentence[i]) 
                    tokens = "/".join(seg_list)
                    tokenizer = tokens.split("/")
                    #print(tokenizer)
                    for j in range(len(char)):
                        if(char[j] in tokenizer):
                            A = char[j]
                            break
                    B=A

                    zh_dependency_parser = StanfordDependencyParser("/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser-3.9.2-models.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/models/chineseFactored.ser.gz")
                    ans = list(zh_dependency_parser.parse(tokens.split("/")))
                    for row in ans[0].triples():
                        if(row[1]=="nsubj" and row[2][0] in char):
                            A = row[2][0]
                            #print("from: "+A)
                        elif(row[1]=="compound:nn" and row[2][0] in char_except):
                            B = row[2][0]
                            #print("to: "+B)
                
                for k in range(len(vocal)):
                    if(vocal[k] in dialogue[i]):
                        detect_print = False
                        break
                
                if(detect_print):
                        print(sentence[i])
                        print(A+": "+dialogue[i])
                else:
                    detect_print = True
                
            # case 1
            else:
                sentence = upper[0]
                dialogue = upper[1].split("」")[0]
                
                seg_list = jieba.cut(sentence) 
                tokens = "/".join(seg_list)
                tokenizer = tokens.split("/")
                #print(tokenizer)
                for j in range(len(char)):
                        if(char[j] in tokenizer):
                            A = char[j]
                            break
                B=A

                zh_dependency_parser = StanfordDependencyParser("/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/jars/stanford-parser-3.9.2-models.jar","/Users/sumeiru/Desktop/StanfordNLP/StanfordNLP/models/chineseFactored.ser.gz")
                ans = list(zh_dependency_parser.parse(tokens.split("/")))
                
                for row in ans[0].triples():
                    if(row[1]=="nsubj" and row[2][0] in char):
                        A = row[2][0]
                        #print("from: "+A)
                    elif(row[1]=="compound:nn" and row[2][0] in char_except):
                        B = row[2][0]
                        #print("to: "+B)

                for k in range(len(vocal)):
                    if(vocal[k] in dialogue):
                        detect_print = False
                        break
                
                if(detect_print):
                        print(A+": "+dialogue)
                else:
                    detect_print = True
                
    else:
        print(x)
        
print("============The End============")
