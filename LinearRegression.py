import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%%Import
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['font.size'] =14
plt.rcParams['font.weight']= 'bold'
plt.style.use('seaborn-whitegrid')

df= pd.read_csv("insurance.csv")

print(df.shape)
sns.lmplot(x="bmi",y="charges",data=df,aspect=2,height=6)
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("BMI vs Charges")
plt.show()

#%%Correlation Matrix

corr = df.corr()
sns.heatmap(corr,cmap="twilight",annot=True)
plt.show()

#%%Dummy Variable Trap

categorical_columns=['sex','children', 'smoker', 'region']
df_encode= pd.get_dummies(data=df,prefix="OHE",prefix_sep="-",columns=categorical_columns,drop_first=True,dtype="int8")

print(df_encode)

#%% Box-Cox Transformation
from scipy.stats import boxcox

y_bc,lam,ci=boxcox(df_encode["charges"],alpha=0.05)
#Not used to df 
#%%Log Transform

df_encode["charges"]=np.log((df_encode["charges"]))


#%%Train test split

from sklearn.model_selection import train_test_split
X=df_encode.drop("charges",axis=1)
y=df_encode["charges"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)

#%%sklearn build model

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)

#%%model evaluation


from sklearn.metrics import mean_squared_error
y_pred_sk=lin_reg.predict(X_test)
J_mse_sk=mean_squared_error(y_pred_sk,y_test)
print("The Mean Square Error(MSE) is " , J_mse_sk)



#%%Model Validation

f,ax = plt.subplots(1,2,figsize=(14,6))
import scipy as sp
_,(_,_,r)= sp.stats.probplot((y_test - y_pred_sk),fit=True,plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')

sns.scatterplot(y = (y_test - y_pred_sk), x= y_pred_sk, ax = ax[1],color='r') 
ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted');