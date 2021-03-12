import pandas as pd
import pickle

df=pd.read_csv('car data.csv')

print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Car_Name'].unique())

df.isnull().sum()



df.drop(['Car_Name'],inplace=True,axis=1)

df['num-of-year']=2020- df['Year']

df.drop(['Year'],axis=1,inplace=True)

df.corr()

df=pd.get_dummies(df,drop_first=True)

X=df.iloc[:,1:]
y=df.iloc[:,0]



from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(9).plot(kind='barh')
plt.show()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()

re=regressor.fit(X_train,y_train)

pred=regressor.predict(X_test)


file = open('selling_price_model.pkl', 'wb')
pickle.dump(regressor, file)