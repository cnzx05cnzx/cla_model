import csv
import pkuseg
import pandas as pd


def new_data():
    data = pd.read_csv('./data/news.csv')
    # data['text'] = data['level_4'].map(str) + data['content'].map(str)
    # df = data[['content', 'label']]

    df = data.rename(columns={'review': 'content', 'sentiment': 'label'})
    df.dropna(axis=0, how='any')
    df['content'] = df['content'].astype('str')
    df['content'] = df['content'].apply(lambda x: x.strip())
    # df['pos'] = df['pos'].astype('str')
    df = df[df['content'].apply(lambda x: len(x) > 1 and x != 'nan')]
    df.to_csv('./data/news.csv', index=0)


def see_data():
    data = pd.read_csv('./data/news.csv')
    data['len'] = data['comment'].apply(lambda x: len(str(x)))
    print(data['len'].describe([.5, .9]))
    print(data['pos'].value_counts())


def make_before():
    data = pd.read_csv('./data/security.csv')
    data = data[data['pos'] == 1]
    file = open(r'./data/before.txt', 'w+', encoding='utf-8')
    for index, row in data.iterrows():
        file.write(str(row['pos']) + '\t' + str(row['word']) + '\n')
    file.close()


def make_after():
    # newline这里是去除行之间的空行
    f = open('./data/security.csv', 'a+', encoding='utf-8', newline="")

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    with open('./data/after.txt', 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for i in lines:
            data = i.split('\t')
            # 因为txt存储数据往往带有"\n"换行，因此在这里需要去掉
            # print(data[0], data[1][:-1])
            csv_writer.writerow([data[1][:-1], data[0]])


if __name__ == '__main__':
    # new_data()
    see_data()
    # make_before()
    # make_after()
