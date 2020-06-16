import pandas as pd
sub = pd.read_csv('xgb_balence.csv')
dd={0:1,1:4,2:5,3:21}
def sub_process(x):
    x=dd[x]
    return x
sub['label']=sub['label'].apply(sub_process)
#a = resT.apply(sum)
#a/3
#b= a/3
#c=pd.DataFrame(b)
sub.to_csv('merger_balence.csv')