import pandas as pd
import pickle
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso



url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv"

df = pd.read_csv(url)

X = df.drop(['CNTY_FIPS','fips','Active Physicians per 100000 Population 2018 (AAMC)','Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)','Active General Surgeons per 100000 Population 2018 (AAMC)','Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)','Total physician assistants (2019)','Total physician assistants (2019)','Total Hospitals (2019)','Internal Medicine Primary Care (2019)','Family Medicine/General Practice Primary Care (2019)','STATE_NAME','COUNTY_NAME','ICU Beds_x','Total Specialist Physicians (2019)'], axis=1)
y = df['ICU Beds_x']


columns = X.columns
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = pd.DataFrame(X_std, columns = columns)
X_std.head()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_std, y, test_size=0.3, random_state=42)

model = LassoCV(cv=5, random_state=0, max_iter=10000)

model.fit(X_train, y_train)

lasso = Lasso(alpha=model.alpha_)

lasso.fit(X_train, y_train)

coef_list=lasso.coef_
loc=[i for i, e in enumerate(coef_list) if e != 0]
col_name=X.columns
print(col_name[loc])


filename = 'models/model.sav'

pickle.dump(lasso, open(filename,'wb'))
