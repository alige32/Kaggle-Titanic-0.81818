import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# 特征预处理map函数
def change_age(x):
    if x < 16:
        return 1
    else:
        return 0


def change_family(x):
    if x == 1 or x == 2 or x == 3:
        return 1
    else:
        return 0


def change_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    elif x == 'Q':
        return 2


def pro_train_data(datas):
    datas['family'] = datas.SibSp + datas.Parch
    datas.loc[[65, 159, 176, 709], 'Age'] = 15
    age = datas.Age.map(change_age)
    datas['Age'] = age
    datas['family'] = datas.family.map(change_family)
    cabin_datas = datas.Cabin.replace(np.nan, 0)
    datas['Cabin'] = np.where(cabin_datas == 0, 0, 1)
    datas.Embarked.fillna('C', inplace=True)
    datas['Embarked'] = datas.Embarked.map(change_embarked)
    datas['Sex'] = np.where(datas['Sex'] == 'female', 1, 0)
    to_drop = ['PassengerId', 'Survived', 'Name', 'Ticket', 'SibSp',
               'Parch']
    target = datas.Survived
    datas_handle = datas.drop(to_drop, axis=1)
    return datas_handle, target


def pro_test_data(test_datas):
    test_datas.loc[(test_datas.Fare.isnull()), 'Fare'] = test_datas[
        (test_datas.Embarked == 'S') & (test_datas.Pclass == 3)].Fare.median()
    test_datas['family'] = test_datas.SibSp + test_datas.Parch
    test_datas.loc[[244, 344, 417], 'Age'] = 15
    age = test_datas.Age.map(change_age)
    test_datas['Age'] = age
    test_datas['family'] = test_datas.family.map(change_family)
    cabin_datas = test_datas.Cabin.replace(np.nan, 0)
    test_datas['Cabin'] = np.where(cabin_datas == 0, 0, 1)
    test_datas.Embarked.fillna('S', inplace=True)
    test_datas['Embarked'] = test_datas.Embarked.map(change_embarked)
    test_datas['Sex'] = np.where(test_datas['Sex'] == 'female', 1, 0)
    to_drop = ['PassengerId', 'Name', 'Ticket', 'SibSp',
               'Parch']
    new_test_data_handle = test_datas.drop(to_drop, axis=1)

    return new_test_data_handle


def clf(datas, label, pre_datas):
    x_train, x_test, y_train, y_test = train_test_split(datas, label, random_state=8)
    xgc = XGBClassifier()
    xgc_param = {
        'n_estimators': range(30, 50, 2),
        'max_depth': range(2, 7, 1)
    }
    gc = GridSearchCV(xgc, param_grid=xgc_param, cv=5)
    gc.fit(x_train, y_train)
    print("训练集样本为：", x_train.shape[0])
    print("测试集样本为：", x_test.shape[0])
    print("预测率为：", gc.score(x_test, y_test))
    print("交叉验证最好结果：", gc.best_score_)
    print("交叉验证最好参数模型：", gc.best_estimator_)
    return gc.predict(pre_datas)


def generate_result(test_data, result):
    result = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': result.astype(np.int32)})
    result.to_csv("demo.csv", index=False)


if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    pro_datas, target = pro_train_data(train_data)
    pre_datas = pro_test_data(test_data)
    pre_y = clf(pro_datas, target, pre_datas)
    generate_result(test_data, pre_y)
