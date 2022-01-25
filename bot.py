import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#データの読み込み&確認
train = pd.read_csv("/Users/e195767/VS code/SIGNATE/bot/train.tsv", sep="\t", index_col=0)
test = pd.read_csv("/Users/e195767/VS code/SIGNATE/bot/test.tsv", sep="\t", index_col=0)
submit = pd.read_csv("/Users/e195767/VS code/SIGNATE/bot/sample_submit.csv", index_col=0, header=None)

print(train.info())
print(train.describe())
res = train.corr()
print(res["bot"])

#可視化
hum_df = train[train["bot"] == 0]
bot_df = train[train["bot"] == 1]

search_idx = "friends_count"

plt.hist(hum_df[search_idx], alpha=0.5, label="human")
plt.hist(bot_df[search_idx], alpha=0.5, label="bot")
plt.show()

features = train.drop("bot", axis=1)
target = train["bot"]

#print(features.shape)
#print(target.shape)

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=0)

#ロジスティック回帰
LR = LogisticRegression()
LR.fit(X_train, y_train)

score = LR.score(X_test, y_test)
#print(score)



pred = LR.predict(test)
pred = pred.astype(np.int64)

submit[1] = pred
submit.to_csv("/Users/e195767/VS code/SIGNATE/bot/submit1.csv", header=None)


#ランダムフォレスト
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)


pred = clf.predict(test)
pred = pred.astype(np.int64)

submit[1] = pred
submit.to_csv("/Users/e195767/VS code/SIGNATE/bot/submit2.csv", header=None)