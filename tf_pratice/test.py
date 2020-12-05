import numpy as np
import pandas as pd

A=pd.read_excel('./A.xlsx',index_col=0)
A=np.array(A,dtype=int)
I=np.identity(13)
R=A+I
R_=R
for i in range(0,13):
    R_=np.matmul(R_,R)
    print(R_)

R_[R_>0]=1

#print(R_)
#R=pd.DataFrame(R_)
#R.to_excel('./R.xlsx')
print(R_)
RR = [val for val in R_ if val in R_.T]
print(RR)
RR=pd.DataFrame(RR)
RR.to_excel('./RR.xlsx')
