from predictor import predict


# predictを自動実行して平均二乗誤差を記録する
def evaluate():
    f = open('result2.txt', 'w')
    conditions = [
        # 比較対象
        {'number': 10000, 'layer': 3},
        # データセットの数を変更
        {'number': 100000, 'layer': 3},
        {'number': 500000, 'layer': 3},
        # 中間層を4層に
        {'number': 10000, 'layer': 4},
        # 中間層を4層にしてデータセットを増やす
        {'number': 500000, 'layer': 4},
    ]
    # functions =['sphere', 'sample']
    functions =['sample']

    for func in functions:
        for con in conditions:
            mae = predict(func, con['number'], con['layer'])
            print(func, con['number'], con['layer'])
            f.write(func + str(con['number']) + str(con['layer']) + '\n')
            print(mae)
            f.write(str(mae) + '\n')

if __name__ == '__main__':
    evaluate()
