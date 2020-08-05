import os
# -*- coding: utf-8 -*-
def del_dir_tree(path):
#''' 递归删除目录及其子目录,　子文件'''
    if os.path.isfile(path):
        try:
            os.remove(path)
            return
        except Exception as e:
            #pass
            print(e)
    elif os.path.isdir(path):
        for item in os.listdir(path):
            itempath = os.path.join(path, item)
            del_dir_tree(itempath)
    try:
        os.rmdir(path) # 删除空目录
    except Exception as e:
        #pass
        print(e)
