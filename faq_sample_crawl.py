import pandas as pd
from random import randrange
from datetime import timedelta
from datetime import datetime
import random, time
from datasets import load_dataset
import json, os, re

out = open("privacy.txt", "w")
for page_index in range(1, 14):
    url = f'https://www.privacy.go.kr/nns/ntc/faq/FaqListInqireList.do?faqId=&pageIndex={page_index}&searchCondition=qestnSj&searchKeyword='
    tables = pd.read_html(url, encoding='utf-8') 
    t = tables[0]
    rows = [] 
    for r in range(len(t)):
        rows.append(t[0][r] + t[1][r])
    if page_index == 5:
        rows[6:11] = []
    for r in range(int(len(rows) / 2)):
        q = rows[r * 2]
        q = re.sub("^질문[\d]+.\s", "", q)
        q = re.sub("열기$", "", q)
        a = rows[r * 2 + 1]
        a = re.sub("^답변\s\so?\s*", "", a)
        out.write(f"Q: {q}\n")    
        out.write(f"A: {a}\n\n")    
out.close()