import xgboost as xgb
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

#todo 数据读取函数
outline_path = "../../IndustrialSteam/data/zhengqi_train.txt"
online_path = "../../IndustrialSteam/data/zhengqi_test.txt"
online_outpath = '../../IndustrialSteam/result/20190412lgb.txt'
def load_pandas(path):
    df = pd.read_csv("../../IndustrialSteam/data/zhengqi_train.txt",sep="\t")
    x = df.drop(['target'],1)
    y = df['target']
    return x,y

#读取完成之后直接可以做训练
def load_libsvm(path):
    dtrain = xgb.DMatrix(path)
    print(dtrain.get_label())

# todo 另外 libsvm sklearn也可以读取
def load_libsvm_sklearn(path):
    from sklearn.datasets import load_svmlight_file
    print(load_svmlight_file(path))

def clean(x,y):
    return 0
#todo 特征工程
def preprocession(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=1)
    # 删除不重要的特征，对预测结果会产生噪声，不能删除太多的数据
    x_train = x_train.drop(['V13','V35'],axis=1)
    x_test = x_test.drop(['V13','V35'],axis=1)
    ## 添加多项式
    ## interaction_only = True：表示仅添加交互项，不添加平方项
    ## include_bias：默认为 True 。如果为 True 的话，那么结果中就会有 0 次幂项，即全为 1 这一列。
    # poly = PolynomialFeatures(degree=2,include_bias=False,interaction_only=True)
    # x_train = poly.fit_transform(x_train)
    # x_test = poly.transform(x_test)
    return x_train,x_test,y_train,y_test

#todo lgb的模型
def lgb_tarin(x_train,x_test,y_train,y_test):
    model = lgb.LGBMRegressor(num_leaves=16,
                     learning_rate=0.1,
                     n_estimators=300,
                     silent=True,
                     reg_alpha=1,
                     reg_lambda=1)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print("train mse:", mean_squared_error(y_train, train_pred))
    print("test mse:", mean_squared_error(y_test, test_pred))
    return model

#todo 1、sklearn
def xgboost_train(x_train,x_test,y_train,y_test):
    model = xgb.XGBRegressor(max_depth=3,
                     learning_rate=0.05,
                     n_estimators=100,
                     silent=True,
                     reg_alpha=1,
                     reg_lambda=1,
                     objective="reg:linear")
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print("train mse:",mean_squared_error(y_train, train_pred))
    print("test mse:",mean_squared_error(y_test, test_pred))
    # 在sklearn中可以查看xgboost模型特征的重要度，从而可以进行特征筛选，清除特征噪声
    fm = model.feature_importances_
    for ids,i in enumerate(fm):
        print("特征是%d,得分为%f"%(ids,i))
    return model

#todo 模型组合多数据进行训练
def combine_model(x_train,x_test,y_train,y_test,C):
    model1 = xgboost_train(x_train,x_test,y_train,y_test)
    model2 = lgb_tarin(x_train,x_test,y_train,y_test)
    from sklearn.linear_model import Ridge #组合模型很容易过拟合，加入一个正则防止过拟合
    y_pred1 = model1.predict(x_train).ravel()
    y_pred2 = model2.predict(x_train).ravel()
    print(y_pred1)
    print(y_pred2)
    secend_x_train = pd.DataFrame({'y_pred1':y_pred1,'y_pred2':y_pred2})
    print(secend_x_train.shape)
    print(y_train.shape)
    linear = Ridge(alpha=C)
    linear.fit(secend_x_train,y_train)
    secend_y_traiin = linear.predict(secend_x_train)
    print("train mse:",mean_squared_error(y_train, secend_y_traiin))

    y_predt1 = model1.predict(x_test).ravel()
    y_predt2 = model2.predict(x_test).ravel()
    secend_x_test = pd.DataFrame({'y_pred1':y_predt1,'y_pred2':y_predt2})
    secend_y_test = linear.predict(secend_x_test)
    print("test mse:",mean_squared_error(y_test, secend_y_test))
    return linear

#todo 1、xgboost原生的训练方式
def xgboost_self(x_train,x_test,y_train,y_test):
    DTrain = xgb.DMatrix(data=x_train,label=y_train) #将数据转换成xgboost模型需要的格式
    DTest = xgb.DMatrix(data=x_test,label=y_test)
    param = {
        "objective":"reg:linear",
        "max_depth":3,
        "eta":0.05,
        "alpha":1,
        "lambda":1,
        "eval_metric":"rmse",
        "random_state":123
    }
    # params：传入模型参数，"objective"这个参数必须一开始就设定，模型属于分类还是回归问题
    # num_boost_round：迭代轮次相当于n_estimators
    # evals = [(DTrain, "train"), (DTest, "test")]：可以自动打印出每轮迭代输出"eval_metric":"rmse"参数定义的均方根误差值，从而判断模型有没有过拟合
    model = xgb.train(params=param,dtrain=DTrain,num_boost_round=100,evals=[(DTrain,"train"),(DTest,"test")])

# todo 导入测试数据并预测，输出结果
def oneline_predict(model,online_path,online_outpath):
    online_test = pd.read_csv(online_path,sep='\t')
    online_test = online_test.drop(['V13' ,'V35'],axis=1)
    online_pred = model.predict(online_test)
    # print('预测值为:',online_pred)
    # with open(online_outpath,'w') as f:
    #     for line in online_pred:
    #         f.write(str(line)+'\n')
    with open(online_outpath,'w') as f:
        online_list=[str(i) for i in online_pred]
        f.write('\n'.join(online_list))

# todo 定义主函数
def main_sklearn():
    x,y = load_pandas(outline_path)
    clean(x,y)
    x_train, x_test, y_train, y_test = preprocession(x,y)
    # xgboost_train(x_train, x_test, y_train, y_test)
    # xgboost_self(x_train, x_test, y_train, y_test)
    # lgb_tarin(x_train, x_test, y_train, y_test)
    # combine_model(x_train,x_test,y_train,y_test, 1)
    # 导出预测数据
    # model=xgboost_train(x_train, x_test, y_train, y_test)
    # model = combine_model(x_train,x_test,y_train,y_test, 1) #组合模型返回的模型预测测试集还有问题，后边有时间修改
    model = lgb_tarin(x_train,x_test,y_train,y_test)
    oneline_predict(model=model, online_path=online_path, online_outpath=online_outpath)

#load_libsvm("./libsvm.txt")
main_sklearn()
