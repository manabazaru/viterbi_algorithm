import numpy as np
import csv

S_REG = 3 # レジスタ数
LENGTH = 259 # 符号長
LT3 = LENGTH * S_REG 
TEST = 100 # テスト回数100000

# 初期化
tdata = rdata = np.zeros((TEST, LENGTH), dtype=np.int)
tcode = rcode = np.zeros((TEST, LT3), dtype=np.int)
transmit = receive = np.zeros((TEST, LT3))
array = [['SNR', 'BER']]
path = __file__ + '/../test.csv'  # CSVの書き込みpath．任意で変えて．

# tdata: 符号化前の送信データ
# tcode: 符号化後の送信データ
# rdata: 復号化前の受信データ
# rcode: 復号化後の受信データ
# transmit: 送信信号
# receive: 受信信号

coding_array = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

def awgn(SNRdB, size):
    # awgnを作る
    
    #信号エネルギーは1とする
    E_s = 1
    
    #符号化率
    S_REG
    R = 1 / S_REG
    
    #雑音N_0
    N_0 = E_s * R * 10 ** (- SNRdB / 10)
    noise = np.random.normal(0, np.sqrt(N_0 / 2), size) \
            + 1j * np.random.normal(0, np.sqrt(N_0/2), size)
    return noise

if __name__ == '__main__':
    # 表示
    print('# SNR BER:')

    # 伝送シミュレーション
    for SNRdB in np.arange(0, 6.25, 0.25):
        # 送信データの生成
        tdata = np.random.randint(0, 2, (TEST, LENGTH - S_REG))
        
        # 終端ビット系列の付加
        end_bit = np.zeros((TEST, S_REG), dtype=np.int)
        tdata = np.c_[tdata, end_bit]

        # 畳み込み符号化
        m, n = tdata.shape
        for i in range(m):
            for j in range(n):
                if j < S_REG:
                    recent_array = np.concatenate([np.zeros(S_REG - j, dtype=np.int), tdata[i, 0:j+1]])
                else:
                    recent_array = tdata[i, j-S_REG:j+1]

                coded_array = np.dot(recent_array, coding_array) % 2
                for col_num in range(S_REG):
                    tcode[i, j*S_REG+col_num] = coded_array[col_num]
        
        # BPSK変調
        transmit[tcode == 0] = -1
        transmit[tcode == 1] = 1

        # 伝送
        receive = transmit + awgn(SNRdB, (TEST, LT3))

        # BPSK復調
        rcode[receive < 0] = 0
        rcode[receive >= 0] = 1

        # ビタビ復号

        # 誤り回数計算
        rdata = tdata
        ok = np.count_nonzero(rdata == tdata)
        error = rdata.size - ok

        # BER計算
        BER = error / (ok + error)

        # 結果表示
        print('SNR: {0:.2f}, BER: {1:.4e}'.format(SNRdB, BER))

        # CSV書き込み．コメントアウト解除すれば書き込める
        array.append([SNRdB, BER])
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(array)