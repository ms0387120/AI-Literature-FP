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
f = open("小毛驢.txt", "r")

#==========抽取元素==========
## NN_Default
NN_except=['個','誰','時','兩','一','房子']
vocal = ['砰','咚','哞','汪','喵','吱','嘖','格','呱','嘩','叮','咚','霹靂啪啦','窸窣','咕','嗤','咩']

loc = ['古埃及','開羅城','清真寺','補鞋攤','房舍','洞房','王國','地窖','農場','市政廳', '水塘', '遊樂園', '圳溝', '池塘', '高塔', '古堡', '噴泉', '山林', '屋頂', '林區', '廚房', '家', '迷宮', '摩天樓', '磚屋', '巖洞', '客廳', '浴缸', '奶奶家', '壩', '六家', '頂層', '座', '山溝', '起居室', '坑', '山巔', '木屋子', '臥室', '修道院', '井里', '花園', '浴室', '摩天大樓', '教堂', '溫莎堡', '莊園', '山腳', '山頂', '佛塔', '聖母院', '庭園', '溪谷', '聖殿', '水池', '主教座堂', '山坡', '竹林', '山崗', '天主教堂', '八仙山', '深潭', '森林', '白金漢宮', '咖啡館', '臥房', '林木', '寺院', '雨林', '王宮', '潭', '塘', '摩天輪', '小河', '斷崖', '電視塔', '宮殿', '溝', '山丘', '庭院', '池', '庴', '稻草屋', '城堡', '大殿', '水潭', '河岸', '夏宮', '儲藏室', '山', '皇宮', '湖畔', '清真寺', '拱門', '原始林', '溪', '森林區', '圓頂', '禮拜堂', '廁所', '大雪山', '河', '崖']
# 老師地點暗示放這

char_except = ['魯夫','強盜','惡婆','馬','老婆','肯納凡','水晶棺','鵝毛大雪','親愛的','王妃','人','丈夫','狼說','好孩子','身邊說','人們心中', '父親','母親','公主','美好願望', '好幾', '兄弟','個個', '小女兒', '美如天仙','小公主','嚴冬時節', '大雪片', '窗子邊', '針線活兒', '寒風捲著', '捲著雪片', '烏木窗臺', '針口流了', '女兒','三點血滴', '飄進窗子', '鮮紅血滴', '烏木窗臺', '血一樣', '這窗子','他們','你們','我們','弟弟','女水妖','斧子','老太婆','說']
family_member= ['兄嫂','祖母','兄妹','前妻', '大伯','前夫','媽咪','伯伯','後娘','姐姐', '伯父', '繼父', '姊妹', '外公', '叔叔','姐弟', '堂兄弟','奶媽', '大哥','小妹', '大姊','老爸','舅舅','外婆','爸媽','大姐','公公', '太太','姪女','二哥', '姨丈','姐妹','祖父','爺爺','阿公','岳父','姊姊','孫兒','堂弟', '嬸婆','外祖母','阿姨', '嬸嬸', '三哥','姐夫', '姑姑','哥哥','大弟', '繼母','弟弟','兄長','後母', '老伴', '姨媽','大嫂','外祖父','曾祖父','岳母', '姊弟','妹妹','奶奶','奶奶','爸爸','姊夫']
passerby = ['馬魯夫','法蒂瑪','磨坊老闆','老山羊','老國王','驢兒','上帝','樂師','農婦','農夫','羊','白雪公主', '副首相', '愛麗絲', '白熊', '老太太', '青蛙', '馬格麗特', '蝙蝠', '趙自強', '王儲', '天鵝', '巫婆', '丫頭', '李訥', '結拜', '馬伕', '王子', '老奶奶', '松鼠', '牧童', '灰姑娘', '畢蘭德拉', '皇后', '妖精', '安妮', '審議官', '兔子', '狐狸', '魔鏡', '蘇菲亞', '新王', '哈山', '怪物', '安娜', '蚱蜢', '武籐嘉文', '魔女', '寶蓮燈', '西遊記', '嬰', '瑪麗亞', '瑪麗', '山豬', '服侍', '幽魂', '孿生', '星星', '夫家', '美人魚', '小妹妹', '稻草人', '女王', '鸚鵡', '莎拉', '經濟院', '碗公', '憶樺', '蜻蜓', '蟒蛇', '愛神', '獅子王', '飛鼠', '烏龜', '小丑', '雙胞胎', '國王', '艾斯瓦利亞', '鐵扇', '菲立普', '夢見', '胡桃鉗', '戽斗', '威廉', '仙履奇緣', '包杜恩', '外相', '藏相', '查理', '醉俠', '妹子', '捕頭', '女巫', '額賀', '楊光南', '幫主', '袋鼠', '僕人', '刺字', '蘿絲', '丸子', '摩納哥', '薔薇', '角頭', '奇遇記', '暱稱', '崇拜者', '俠盜', '仙女', '烏賊', '飛天', '同居人', '貓頭鷹', '李素華', '阿布杜拉', '烏鴉', '吳火眼', '郭朝明', '女僕', '前首相', '雲雀', '阿拉丁', '金枝玉葉', '皇子', '星願', '次官', '老婆婆', '精靈', '老大', '史蒂芬妮', '甜甜', '瑪格麗特', '王后', '詩琳通', '救世主', '親生', '白鵝', '親王', '阿嬤', '珍妃', '小弟', '貓咪', '野狼', '駱駝', '花木蘭', '大臣', '獵人', '猴子', '吸血鬼', '貓', '露西', '巴爾札克', '小白', '哈利', '雷尼爾', '賈南德拉', '雙親', '野豬', '吳仕傑']
# 老師人物暗示放這

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
f.close()

#==========雙引號轉成上下引號==========
rf=open("小毛驢.txt","r")
wf=open("小毛驢_1.txt","w")
quote=0

for x in rf:
    #print(x)
    #print(quote)
    x = x.replace("\t","")
    x = x.replace("\n","")
    if('\"' in x):
        for q in range(len(x)):
            if(quote==0 and x[q]=='\"'):
                y = x[:q] + '「' + x[q+1:]
                quote=1
            elif(quote==1 and x[q]=='\"'):
                y = y[:q] + '」' + y[q+1:]
                quote=0
        wf.write(y+"\n")
    elif('：' in x and '\"' not in x):
        for q in range(len(x)):
            if(x[q]=='：' and x[q+1]!='\"'):
                y = x[:q] + '「' + x[q+1:len(x)-1]+'」'
        wf.write(y+"\n")
    elif(x!="\n"):
        wf.write(x+"\n")
        
    quote=0

rf.close()
wf.close()
print("Quote Transform Finished")

#==========說話人辨識==========
import random

f = open("小毛驢_1.txt","r")
wf = open("小毛驢_2.txt","w")
prev_occur = []
occur_char=[]
last_speaker=""
#print(char)

# initialize
for i in range(len(char)):
    prev_occur.append(0)

for x in f:
    #print(prev_occur)
    if(x[0]=="「" and x[-2]=="」"):
        
        for j in range(len(prev_occur)):
            if(prev_occur[j]!=0):
                occur_char.append(char[j])
        
        #print(occur_char)
        if(len(occur_char)>1):
            speaker = random.choice(occur_char)
            while(last_speaker==speaker):
                speaker = random.choice(occur_char)
        
        else:
            #主角
            occur_char.append(char[0])
            speaker = random.choice(occur_char)
            while(last_speaker==speaker):
                speaker = random.choice(occur_char)

        print(x[:-1]+speaker+"說\n")
        wf.write(x[:-1]+speaker+"說\n")
        last_speaker = speaker
        occur_char=[]
    
    #other types
    elif(x!="\n"):
        
        wf.write(x)
        for i in range(len(prev_occur)):
            prev_occur[i]=0
            
        for j in range(len(prev_occur)):
            if(char[j] in x):
                    prev_occur[j]+=1
    
f.close()
wf.close()
print("Speaker Recognition Finished")

#==========判斷場景切割法==========
### Input reading
f = open("小毛驢_2.txt", "r")
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

f.close()
#==========生成劇本==========
# Latest Version

### Input reading
f = open("小毛驢_2.txt", "r")
wf = open("小毛驢_劇本.txt", "w")

last_loc = ""
last_time=""
start=True
detect_print=True

#老師劇本名稱放這
wf.write("劇本名稱：小毛驢\n")

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
                    wf.write("================================"+"\n")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    wf.write("Scene: 時/ "+time+"     景/ "+loc_set[i]+"\n")
                    last_loc = loc_set[i]
                    last_time = time
                    start=False
                    break
                
                if(start==True):
                    print("================================")
                    wf.write("================================"+"\n")
                    last_loc = '家'
                    print("Scene: 時/ "+time+"     景/ "+last_loc)
                    wf.write("Scene: 時/ "+time+"     景/ "+last_loc+"\n")
                    start = False
                    break
            
            # 地點 case2: 家以外的場景
                if(loc_set[i] in x and loc_set[i]!=last_loc and loc_set[i]!="家"):
                    print("================================")
                    wf.write("================================"+"\n")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    wf.write("Scene: 時/ "+time+"     景/ "+loc_set[i]+"\n")
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
                    wf.write("================================"+"\n")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    wf.write("Scene: 時/ "+time+"     景/ "+loc_set[i]+"\n")
                    last_loc = loc_set[i]
                    last_time = time
                    start=False
                    break
                
                if(start==True):
                    print("================================")
                    wf.write("================================"+"\n")
                    last_loc = '家'
                    print("Scene: 時/ "+time+"     景/ "+last_loc)
                    wf.write("Scene: 時/ "+time+"     景/ "+last_loc+"\n")
                    start = False
                    break
            
            # 地點 case2: 家以外的場景
                if(loc_set[i] in x and loc_set[i]!=last_loc and loc_set[i]!="家"):
                    print("================================")
                    wf.write("================================"+"\n")
                    print("Scene: 時/ "+time+"     景/ "+loc_set[i])
                    wf.write("Scene: 時/ "+time+"     景/ "+loc_set[i]+"\n")
                    last_loc = loc_set[i]
                    last_time = time
                    break
                
                '''if(time!=last_time):
                    print("================================")
                    print("Scene: 時/ "+time+"     景/ "+last_loc)
                    last_time = time'''
            
    
        # 強迫換景
        if('隔天' in x or '第二天' in x or '第三天' in x):
            print("================================")
            wf.write("================================"+"\n")
            print("Scene: 時/ "+time+"     景/ "+last_loc)
            wf.write("Scene: 時/ "+time+"     景/ "+last_loc+"\n")
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
                    wf.write(A+": "+dialogue[i]+"\n")
                    print(sentence[i])
                    wf.write(sentence[i]+"\n")
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
                        wf.write(A+": "+dialogue+"\n")
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
                        wf.write(sentence[i]+"\n")
                        print(A+": "+dialogue[i])
                        wf.write(A+": "+dialogue[i]+"\n")
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
                        wf.write(A+": "+dialogue+"\n")
                else:
                    detect_print = True
                
    else:
        print(x)
        wf.write(x+"\n")
        
print("============The End============")
wf.write("============The End============"+"\n")
f.close()
wf.close()
