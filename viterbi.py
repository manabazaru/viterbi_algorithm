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

#ビタビ復号用クラス
#ノードをつなぐ矢印のクラス
class Arrow():
    def __init__(self, ip, op, origin):
        self.input = ip     #符号器へ入力された信号(1 or 0)
        self.output = op    #1×S_REGのビット群配列
        self.origin = origin #ノードをつなぐ矢印の根本ノード. S_000が0

#結果用矢印のクラス
class ResultArrow(Arrow):
    def __init__(self, ip, op, origin, result):
        super().__init__(ip, op, origin)
        self.result = result

#状態ノードクラス
class Node():
    def __init__(self, node_num, arrow1, arrow2):
        self.num = node_num
        self.arrows = [arrow1, arrow2]
    
    def calc_min_cost(self, layer, node_array, rcv_sig):
        rlt_array = [-1, -1]    #2つの矢印の結果を格納する変数
        for i in range(2):      #2つの矢印をそれぞれ計算
            arrow = self.arrows[i]
            if node_array[layer-1][arrow.origin] == -1: #前のノードがない場合, 結果はない
                continue
            xor_array = rcv_sig ^ arrow.output      #ビタビ復号器のレジスタビットと受信信号のxor
            ham_dis = np.sum(xor_array) + node_array[layer-1][arrow.origin]             #ノードまでの矢印ごとのハミング距離
            rlt_array[i] = ham_dis      #ハミング距離を結果に登録
        if rlt_array[0] == -1:          #arrow1の前ノードがない or 両方ない :arrow2
            rlt = rlt_array[1]
            rlt_arrow = self.arrows[1]
        elif rlt_array[1] == -1:        #arrow2の前ノードがない :arrow1
            rlt = rlt_array[0]
            rlt_arrow = self.arrows[0]
        else:                           #両方存在
            if rlt_array[0] < rlt_array[1]: #arrow1の方がハミング距離が小さい: arrow1
                rlt = rlt_array[0]
                rlt_arrow = self.arrows[0]
            else:                           #arrow2の方がハミング距離が小さい or 同じ: arrow2
                rlt = rlt_array[1]
                rlt_arrow = self.arrows[1]
        ret_arrow = ResultArrow(rlt_arrow.input, rlt_arrow.output, rlt_arrow.origin, rlt)
        return ret_arrow

#ノードの作成(ノード番号は000, 001, ...の順に0, 1, ...と割り振る)
#ノードに対応する矢印を作成
#ノード0
ar00 = Arrow(0, np.array([0, 0, 0]), 0)
ar01 = Arrow(0, np.array([1, 1, 1]), 4)
node0 = Node(0, ar00, ar01)
#ノード1
ar10 = Arrow(1, np.array([1, 1, 1]), 0)
ar11 = Arrow(1, np.array([0, 0, 0]), 4)
node1 = Node(1, ar10, ar11)
#ノード2
ar20 = Arrow(0, np.array([0, 1, 1]), 1)
ar21 = Arrow(0, np.array([1, 0, 0]), 5)
node2 = Node(2, ar20, ar21)
#ノード3
ar30 = Arrow(1, np.array([1, 0, 0]), 1)
ar31 = Arrow(1, np.array([0, 1, 1]), 5)
node3 = Node(3, ar30, ar31)
#ノード4
ar40 = Arrow(0, np.array([1, 0, 1]), 2)
ar41 = Arrow(0, np.array([0, 1, 0]), 6)
node4 = Node(4, ar40, ar41)
#ノード5
ar50 = Arrow(1, np.array([0, 1, 0]), 2)
ar51 = Arrow(1, np.array([1, 0, 1]), 6)
node5 = Node(5, ar50, ar51)
#ノード6
ar60 = Arrow(0, np.array([1, 1, 0]), 3)
ar61 = Arrow(0, np.array([0, 0, 1]), 7)
node6 = Node(6, ar60, ar61)
#ノード7
ar70 = Arrow(1, np.array([0, 0, 1]), 3)
ar71 = Arrow(1, np.array([1, 1, 0]), 7)
node7 = Node(7, ar70, ar71)

#ノードの配列
nodes = [node0, node1, node2, node3, node4, node5, node6, node7]

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
        rrow, rcol = rcode.shape
        p_num = int(rcol/S_REG)
        nodes_num = 2 ** S_REG
        for sig_num in range(rrow):
            n_cost_array = np.full((p_num+1, nodes_num), -1) #ノードのコスト配列
            n_cost_array[0, 0] = 0                          #最初の000のみ0で初期化
            p_origin_array = np.full((p_num, nodes_num), -1) #ノードにつながる矢印の根本を格納する配列
            p_bit_array = np.full((p_num, nodes_num), -1)    #ノードにつながる矢印のビット(0or1)を格納する配列
            for layer in range(1, p_num):
                chead = (layer - 1) * S_REG
                rcv_sig_ele = rcode[sig_num, chead:chead+S_REG]
                for node_num in range(nodes_num):
                    node = nodes[node_num]
                    best_arrow = node.calc_min_cost(layer, n_cost_array, rcv_sig_ele)
                    n_cost_array[layer, node_num] = best_arrow.result
                    p_origin_array = best_arrow.origin
                    p_bit_array = best_arrow.input
            #トレリス線図の結果生成に必要な初期化
            rlt_ptr = [p_num-1, 0]

            
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