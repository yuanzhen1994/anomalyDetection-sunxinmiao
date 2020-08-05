import os
import numpy as np
import copy
import unzip
import myfile

#文件路径
CurrentDir = os.getcwd()
NormalDatasetDir = r'../stidetraces';
AbnormalDatasetDir = r'../stide-intrusion';
TempDir = r'../stidetraces_work';


print('正在解压缩文件..')
unzip.un_gz(r'../stidetraces.tar', r'../stidetraces.tar.gz')
unzip.un_tar(NormalDatasetDir, r'../stidetraces.tar')
InfoList = os.listdir(NormalDatasetDir+'/Int');
for info in InfoList:
    info_dest = info.replace('.gz', '')
    unzip.un_gz(NormalDatasetDir+'/'+info_dest, NormalDatasetDir+'/Int/'+info)
myfile.del_dir_tree(r'../stidetraces.tar')
myfile.del_dir_tree(NormalDatasetDir+'/Int')

if os.path.isdir(AbnormalDatasetDir):
    pass
else:
    os.mkdir(AbnormalDatasetDir)
unzip.un_gz(AbnormalDatasetDir+'/init', r'../stide-intrusion.gz')

print('OK')


#读取源文件
print('正在解析源文件..')
InfoList = os.listdir(NormalDatasetDir);
NormalTracesPID = [];
NormalTracesDataSet = [];
for info in InfoList:
    temp = [];
    f = open(NormalDatasetDir+'/'+info, 'r');
    for line in f.readlines():
        if line=='\n':
            continue;
        a,b = map(int, line.split());
        temp.append(b);
    NormalTracesPID.append(a);
    NormalTracesDataSet.append(temp);
    f.close();
    
InfoList = os.listdir(AbnormalDatasetDir);
AbnormalTracesPID = [];
AbnormalTracesDataSet = [];
a_last = -1;
temp = [];
f = open(AbnormalDatasetDir+'/'+InfoList[0], 'r');
for line in f.readlines():
    if line=='\n':
        continue;
    a,b = map(int, line.split());
    if a != a_last:
        if temp != [] and a_last != -1:
            AbnormalTracesPID.append(a_last);
            AbnormalTracesDataSet.append(temp);
        temp = [];
        a_last = a;
    temp.append(b);
if temp != [] and a_last != -1:
    AbnormalTracesPID.append(a_last);
    AbnormalTracesDataSet.append(temp);
f.close();
print('OK')


print('正在分析并粗处理数据集..')
if os.path.isdir(TempDir):
    pass
else:
    os.mkdir(TempDir)
np.save(TempDir+'/NormalPID.npy', np.array(NormalTracesPID))
np.save(TempDir+'/NormalData.npy', np.array(NormalTracesDataSet))
np.save(TempDir+'/AbnormalPID.npy', np.array(AbnormalTracesPID))
np.save(TempDir+'/AbnormalData.npy', np.array(AbnormalTracesDataSet))
#mm1=list(np.load(TempDir+'/NormalPID.npy'))
#mm2=list(np.load(TempDir+'/NormalData.npy'))


#删除连续重复的事件，单个事件连续重复的次数最多为 x
def delete_continued_repeated_event(Dataset, x):
    for trace in Dataset:
        i = 0; #the index of trace
        j = 0; #单个事件连续重复的次数
        n = len(trace);
        tl = -1;
        while(n > i):
            if trace[i] == tl:
                j+=1;
                if j>=x:
                    del(trace[i]);
                    i-=1;
                    j-=1;
                    n-=1;
            else:
                j=0;
            tl = trace[i];
            i+=1;
        break;

NormalTracesDataSet_r2 = copy.deepcopy(NormalTracesDataSet);
delete_continued_repeated_event(NormalTracesDataSet_r2, 2);
np.save(TempDir+'/NormalData_r2.npy', np.array(NormalTracesDataSet_r2))

NormalTracesDataSet_r1 = copy.deepcopy(NormalTracesDataSet_r2);
delete_continued_repeated_event(NormalTracesDataSet_r1, 1);
np.save(TempDir+'/NormalData_r1.npy', np.array(NormalTracesDataSet_r1))   

AbnormalTracesDataSet_r2 = copy.deepcopy(AbnormalTracesDataSet);
delete_continued_repeated_event(AbnormalTracesDataSet_r2, 2);
np.save(TempDir+'/AbnormalData_r2.npy', np.array(AbnormalTracesDataSet_r2))

AbnormalTracesDataSet_r1 = copy.deepcopy(AbnormalTracesDataSet_r2);
delete_continued_repeated_event(AbnormalTracesDataSet_r1, 1);
np.save(TempDir+'/AbnormalData_r1.npy', np.array(AbnormalTracesDataSet_r1))  

print('OK')


print('处理完成，解析结果保存于下列文件夹中：')
print(os.path.abspath(TempDir))

NormalTracesDataSet_r2=list(np.load(TempDir+'/NormalData_r2.npy'))
NormalTracesDataSet_r1=list(np.load(TempDir+'/NormalData_r1.npy')) 
AbnormalTracesDataSet_r2=list(np.load(TempDir+'/AbnormalData_r2.npy'))
AbnormalTracesDataSet_r1=list(np.load(TempDir+'/AbnormalData_r1.npy')) 
            

