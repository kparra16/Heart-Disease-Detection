#----------------------------------------------------
# 1. Descriptive Analysis
#----------------------------------------------------

import pandas as pd
import numpy as np
import random as rand

heart=pd.DataFrame(pd.read_excel('D:\\Mis Documentos\\Data Science Certificate\\Group Assigment\\Karen\\heart.xlsx',sep=',',header=0,na_values='NaN'))

heart

#heart.dtypes

heart2=heart[['age','trestbps','chol','thalach','oldpeak']]

heart2.describe()

heart2.skew()


heart['ageGroup'] = ['Youth' if age < 19 else 'Senior' if age > 60 else 'Adult' for age in heart['age']]
heart['OldPeak2'] = ['Lower' if x < 0.8 else 'Higher' for x in heart['oldpeak']]


#Additional grouping from research
#assuming resting blood pressure is diastolic blood pressure. based on https://www.webmd.com/hypertension-high-blood-pressure/guide/diastolic-and-systolic-blood-pressure-know-your-numbers#1-3
heart['RestBloodPressure'] = ['High' if x < 120 else 'Hypertensive Crisis' for x in heart['trestbps']]
heart['Cholestoral'] = ['Normal' if x < 130 else 'High' for x in heart['chol']]
def calc_maxHeartRate (num):
    age, maxrate = num
    return 'Normal' if maxrate <= 220 -age else 'High'
heart['MaxHeartRate'] = heart[['age', 'thalach']].apply(calc_maxHeartRate, axis= 1)

#Source of information: https://litfl.com/st-segment-ecg-library/

heart['propension']=heart[['age', 'thalach']].apply(lambda x: 1 if (x['age']>=58) & (x['thalach']>0)  else 0, axis=1)




#----------------------------------------------------
# 2. Selection of the variables
#----------------------------------------------------

#*--For numerical variables--*
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

heart2.corr()

plt.matshow(heart2.corr())
plt.xticks(range(len(heart2.columns)), heart2.columns)
plt.yticks(range(len(heart2.columns)), heart2.columns)
plt.colorbar()
plt.show()

decil=heart2.quantile(np.arange(0,1.1,0.1),axis=0)             #VER TEMA DE DECILES CON VARIABLE OLDPEAK


vec=[]
count1=[]
count0=[]
count_tot=[]

#Sum 1
sum1=np.sum(heart['target'])

#Sum 0 
sum0=heart.shape[0]-np.sum(heart['target'])




for k in decil.columns:

       if len(decil[k])==decil.shape[0]:
              for i in range(0,len(decil[k])-1):
                     if i+1==(len(decil[k])-1):
                            count1.append([k,str(decil[k].iloc[i])+'-'+str(decil[k].iloc[i+1]),float(heart[(heart[k]>=(decil[k].iloc[i])) & (heart[k]<=(decil[k].iloc[i+1]))].agg({'target':np.sum})),heart[(heart[k]>=(decil[k].iloc[i])) & (heart[k]<=(decil[k].iloc[i+1]))].shape[0]]) 
                     else:
                            count1.append([k,str(decil[k].iloc[i])+'-'+str(decil[k].iloc[i+1]),float(heart[(heart[k]>=(decil[k].iloc[i])) & (heart[k]<(decil[k].iloc[i+1]))].agg({'target':np.sum})),heart[(heart[k]>=(decil[k].iloc[i])) & (heart[k]<(decil[k].iloc[i+1]))].shape[0]]) 
                            

       else:
              for i in range(0,len(decil[k])-1):

                     if i+1==len(decil[k])-1:
                            count1.append([k,str(decil[k].iloc[i])+'-'+str(decil[k].iloc[i+1]),float(heart[(heart[k]>=(decil[k][i])) & (heart[k]<=(decil[k][i+1]))].agg({'target':np.sum})),heart[(heart[k]>=(decil[k][i])) & (heart[k]<=(decil[k][i+1]))].shape[0]]) 
                     else:
                            count1.append([k,str(decil[k].iloc[i])+'-'+str(decil[k].iloc[i+1]),float(heart[(heart[k]>=(decil[k][i])) & (heart[k]<(decil[k][i+1]))].agg({'target':np.sum})),heart[(heart[k]>=(decil[k][i])) & (heart[k]<(decil[k][i+1]))].shape[0]]) 






#count1_pd.to_excel("D:\Mis Documentos\Data Science Certificate\Group Assigment\Karen\heart_deciles.xlsx")


#*--For categorical variables--*

vec2=[]
heart4=[]

heart3=heart[['sex','cp','fbs','restecg','exang','slope','ca','ageGroup','OldPeak2','RestBloodPressure','Cholestoral','MaxHeartRate','propension']]
k='Cholestoral'

list_cat=[]
for k in heart3.columns:
       heart4=pd.concat([heart.groupby(k)['target'].count(),heart.groupby(k).aggregate({'target':np.sum})],axis=1)
       temp_work=pd.DataFrame(heart4.index)
       temp_work.set_index(heart4.index,inplace=True)
       heart4_1=pd.concat([heart4,temp_work],axis=1)
       heart4_1['Variable']=k
       heart4_1.columns
       heart4_1.columns=['target', 'target', 'Levels', 'Variable']
       heart4_1.index
       # if k!=heart3.columns[0]:
       #        hearts_concatenated=pd.concat([heart4_1,hearts_concatenated],axis=0)
       # else :
       #        hearts_concatenated=heart4_1

       list_cat.append(heart4_1)
       print(heart4_1)
print(hearts_concatenated)


for k in range(0,len(list_cat)):
       list_cat[k].columns=['Total','Count1','Levels','Variable']
       list_cat[k]['Count0']=list_cat[k]['Total']-list_cat[k]['Count1']
       list_cat[k]['P_count1']=list_cat[k]['Count1']/sum1
       list_cat[k]['P_count0']=list_cat[k]['Count0']/sum0
       list_cat[k]['Dif']=abs(list_cat[k]['P_count1']-list_cat[k]['P_count0'])
       list_cat[k]['KS']=max(list_cat[k]['Dif'])


#Selected variables: 'oldpeak','ca','cp','exang','propension','sex','slope'


#*---Standarization of variables---#

import math

#Selecting only the variables with Decile with values (excluding 0 partitions)
oldpeak3=count1_pd[count1_pd['Variable']=='oldpeak'].iloc[3:10]

def ln(counts):
       count1,count0=counts
       return math.log10(count1/count0)


oldpeak3['ln']=oldpeak3[['count1','count0']].apply(ln,axis=1)

oldpeak3.shape[0]

heart['oldpeak'][heart['oldpeak']>=float(oldpeak3_2[i])]
print(oldpeak3)
oldpeak3['Decile'].iloc[6]
heart2['LN_oldpeak']=heart2['oldpeak']
for u in range(0,oldpeak3.shape[0]):
      print (u)
      oldpeak3_2=oldpeak3['Decile'].iloc[u].split(sep='-')
      print("esto es olpeak3_2[0] %1.2f y olpeak3_2[1] %1.2f " % (float(oldpeak3_2[0]),float(oldpeak3_2[1])))
      
      if u!=oldpeak3.shape[0]-1:
             heart2['LN_oldpeak'][(heart['oldpeak']>=float(oldpeak3_2[0])) & (heart['oldpeak']<float(oldpeak3_2[1]))]=oldpeak3['ln'].iloc[u]
      else:
             heart2['LN_oldpeak'][(heart['oldpeak']>=float(oldpeak3_2[0])) & (heart['oldpeak']<=float(oldpeak3_2[1]))]=oldpeak3['ln'].iloc[u]
#       print(heart['oldpeak'][(heart['oldpeak']>=float(oldpeak3_2[0])) & (heart['oldpeak']<float(oldpeak3_2[1]))])
      
#       (heart['oldpeak']>=float(oldpeak3_2[i])) and (heart['oldpeak']<float(oldpeak3_2[i+1])):
#              heart['ln_oldpeak']=oldpeak3['ln']

print(heart2[['LN_oldpeak','oldpeak']])     

#For categorical variables#

cat_select=['ca','cp','exang','propension','sex','slope']

counter=0
list_cat2=[]
for k in range(0,len(list_cat)):
       if list_cat[k].index.name in cat_select :
           list_cat[k]['LN']=list_cat[k][['Count1','Count0']].apply(ln,axis=1)
           print(list_cat[k][['Levels','Variable','LN']])
           list_cat2.append(list_cat[k][['Levels','Variable','LN']])


list_cat3=[]
for j in range(0,len(list_cat2)) :
       
       data_base1=pd.merge(heart[list_cat2[j].index.name],list_cat2[j][['LN','Levels']],how='left',left_on=list_cat2[j].index.name, right_on='Levels')
       data_base1.columns=[list_cat2[j].index.name,"".join(['LN_',list_cat2[j].index.name]),'Levels']
       list_cat3.append(data_base1)
       
list_cat3       


temp=np.array(np.zeros(heart.shape[0]))

for h in range(0,len(list_cat3)):
       if h!=0:
              temp=pd.concat([list_cat3[h].iloc[:,1],pd.DataFrame(temp)],axis=1)
       else:
              temp=pd.DataFrame(list_cat3[h].iloc[:,1])              


temp.shape[0]
heart2.shape[0]
heart.columns
temp.columns=['sex','cp','exang','slope','ca','propension']

heart5=pd.concat([temp,heart['id'],heart['target']],axis=1)

#Normalized variables

heart6=pd.concat([heart5,heart2['LN_oldpeak']],axis=1) 

heart6



#----------------------------------------------------
# 3. Train, Test samples. Fitting the model
#----------------------------------------------------

heart6['Random']=  heart6['id'].apply(lambda x: rand.random())
heart6.columns

heart6=heart6[['LN_propension', 'LN_ca', 'LN_slope', 'LN_exang', 'LN_cp', 'LN_sex','LN_oldpeak','id', 'target','Random']]

heart6=heart6.sort_values('Random',ascending=True)

heart7=np.array(heart6.iloc[:,range(0,6)])

heart8=np.array(heart6['target'].values)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(heart7,heart8, test_size = 0.3) 

#*---Fitting and predict the values---*


from sklearn.linear_model import LogisticRegression

glm=LogisticRegression(solver='saga')

glm.fit(X_train,Y_train)

y_pred=glm.predict(X_test)

accuracy=np.sum(Y_test==y_pred)/Y_test.shape[0]