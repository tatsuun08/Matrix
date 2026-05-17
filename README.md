# myMatrix
このプロジェクトはOpenMP・SIMDによる行列演算の高速化を通して，CPU最適化について学ぶ

## 最適化概要
* **メモリアクセス**　メモリが連続に並んでいることによるメモリアクセスの最適化（空間的局所性）
* **マルチスレッド**　OpenMPを用いることにより，1つのコアによる処理ではなく，複数コアで計算を高速化
* **SIMD**  特にAVX2を用いることで機械命令でスカラー処理をするのではなく，まとまったベクトル処理をすることにより高速化
* **キャッシュブロッキング** 一度計算で用いた値はしばらくの間覚えているという特性（時間的局所性）を用いることでメモリアクセスを削減

## プログラムを段階的に改善する
入力される行列をA(row=N, column=K), B(K, M)と設定<br>
関数の実行から終了までの時間を測定する
<br>
M=4096, K=4096, N=4096

<details open><summary> **mul_mat_0** メモリアクセスの悪いプログラム</summary>
このプログラムではループをi,j,k(0 <= i< N, 0 <= j < M, 0 <= k < K)の順で回していきます
<br>
このプログラムの悪いところは行列Bのメモリアクセスが悪いことです．
<br><br>
例1)　M=10という仮設定します<br>
j=0, k=0とするとB[0]にアクセスします<br>
ループが更新されたとき値が変わるのは内側のループであるkが優先です<br>
そのためj=0, k=1, B[10]にアクセスすることになり，B[0]から9個離れたアドレスに飛んでいます<br>
コンピュータとしては，データをB[0], B[1], B[2]..,B[10]と探していくため，データを探す時間が発生します．<br>
→mul_mat_1ではそのメモリアクセスを改善します．

```cpp
for (int i=0; i<N; i++){
    for (int j=0; j<M; j++){
        for (int k=0; k<K; k++){
            result[M * i + j] += A[K * i + k] * B[M * k + j];
        }
    }
}
```
実行時間：20811.59ms
</details>
<details open><summary>mul_mat_1 メモリアクセスの改善</summary>
mul_mat_0のメモリアクセスを改善したプログラム
result[M * i + j] += A[K * i + k] * B[M * k + j]<br>
インデックスは連続に動くようにすればよいので,ループの順序の設定をします．<br>
result[M * i + j] iがjよりも先に動いてしまうと，インデックスが飛んでしまいます．優先度：i < j <br>
A[K * i + k] 同様にして i < k <br> 
B[M * k + j] 同様にして k < j <br>
したがって，ループの順番は i < k < jのように設定することで，メモリアクセスが改善されます．<br>

実行時間： 6240.82ms mul_mat_0よりも3.33倍の高速化
</details><br>

<details open><summary>mul_mat_2 マルチスレッド化</summary>
マルチスレッド化するためにここではOpenMPを用いています．for文の前に#pragma omp parallel forをつけることで，マルチスレッド化が容易に行えます．スレッド数はOMP_NUM_THEREADSを参照します．実よりもでは，Pコア6個，Eコア8個の14個を搭載したCPUで行いました．各コアに計算を等分に割り当てると処理の遅いEコアの処理を待つことになってしまうため，処理が終わったコアに処理を割り当てています（動的スケジューリング）．<br>
実行時間：302.58ms mul_mat_1から20.66倍の高速化
</details><br>

<details open><summary>mul_mat_3 SIMDの導入</summary>
SIMDは，一命令で複数データを同時に扱うことができます．ここでは8個の単精度浮動小数点（float）の計算ができるAVX2を用いています．<br>
※avx2は256ビットを同時に処理でき，float(32bit)　256/32=8 個の計算を1命令で行える<br>
実行時間：173.98ms mul_mat_2より1.73倍高速化
</details><br>

<details open><summary>mul_mat_4 データの詰め替え/summary>
</details><br>
mul_mat_4   155.39ms
<details open><summary>mul_mat_5 キャッシュブロッキング</summary>
mul_mat_5   227.24ms
</details><br>
<details open><summary>mul_mat_6 ループアンローリング</summary>
mul_mat_6   133.69ms
</details><br>
<details open><summary>mul_mat_7 キャッシュブロッキング＋ループアンローリング</summary>
mul_mat_7   148.36ms
</details><br>
<details open><summary>mul_mat_8 GEBPアルゴリズム</summary>
mul_mat_8   214.72ms
</details><br>
