


    nlu_source_choices = ["comment", "comments", "content", "comment_content", "comments_content",
                          'content_comment', 'content_comments']


更改log csv名字
更改myout文件夹名字
更改checkpoint文件夹名字
更换数据

mv log train_history/comment_content/
mv myout.file train_history/comment_content/
mv checkpoints train_history/comment_content/
mv data/MUSIC/index*  train_history/comment_content/
mkdir log
mkdir checkpoints 



mv ft_local/indexed_*    data/MUSIC/
nohup python3 -u train_bert.py  > myout.file 2>&1 &
tail -f myout.file

cat myout.file | grep 'loss='