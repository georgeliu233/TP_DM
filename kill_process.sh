path='/home/haochen/SMARTS/smarts/'
ps -ef| grep $path | grep -v grep|awk '{print "kill -9 " $2}'|bash