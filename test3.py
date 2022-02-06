

import mariadb
import sys

conn = mariadb.connect( 
    user='hunny', 
    password='hunny', 
    host='ec2-3-36-124-21.ap-northeast-2.compute.amazonaws.com', 
    port=3306,
    database='COVID'
)

cur = conn.cursor()

sql = 'insert into mapleSpeaker(game, server,`time`, simbol, nickname, chanel, link, content, origin_data) values("메이플",	"스카니아",	now(),	"testz",	"test",	"test",	"test",	"test",	"origin")'

cur.execute(sql)
conn.commit()
