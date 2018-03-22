import sys
import pickle
import numpy as np

newfeat2id = {'PM2.5':0, 'PM10':1}

def clean_test(inputs=sys.argv[1]):
    dataset = np.empty((260, 9, 2))
    with open(inputs, encoding='BIG5') as f:
        for index, line in enumerate(f):
            index, feat, *val = line.strip().split(',')
            _, real_id = index.split('_')
            real_id = int(real_id)
            if feat == 'RAINFALL' or feat == 'WIND_DIREC':
                continue
            val = [float(i) for i in val]
            if feat in newfeat2id:
                dataset[real_id, :, newfeat2id[feat]] = val
    return dataset


if __name__ == '__main__':
    with open('best.pickle', 'rb') as f:
        lr = pickle.load(f)

    test_dataset = clean_test()

    for index, testcase in enumerate(test_dataset):
        if np.any(testcase[:, 0] > 120) or np.any(testcase[:, 0] < 0):
            need_mod = np.concatenate((np.where(testcase[:, 0] > 120)[0], np.where(testcase[:, 0] < 0)[0]))
            for i in need_mod:
                testcase[i][0] = testcase[i+1][0] if i < 7 else testcase[i-1][0]

    Xt = test_dataset.reshape(-1, 18)
    ans = lr.predict(Xt)

    with open(sys.argv[2], 'w+') as f:
        f.write('id,value\n')
        for i, v in enumerate(ans):
            v = max(min(v, 120), 0)
            f.write('id_%d,%f\n' % (i, v))