#include <cstring>
#include <cstdio>
#include <vector>
#include <chrono>
#include <functional>
#include <random>
#include <algorithm>
#include <string>
#include <immintrin.h>
#include "matrix.hpp"

//NO OPTIMIZE 
Matrix mul_mat_0(const Matrix& A, const Matrix& B){
    int N, K, M;
    N = A.get_row();
    K = A.get_col();
    M = B.get_col();

    assert(K == B.get_row());
    
    std::vector<float> result(N*M);

    for (int i=0; i<N; i++){
        for (int j=0; j<M; j++){
            for (int k=0; k<K; k++){
                result[M * i + j] += A[K * i + k] * B[M * k + j];
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

//Improve memory accece
Matrix mul_mat_1(const Matrix& A, const Matrix& B){
    int N, K, M;
    N = A.get_row();
    K = A.get_col();
    M = B.get_col();

    assert(K == B.get_row());
    
    std::vector<float> result(M * N);

    for (int i=0; i<N; i++){
        for (int k=0; k<K; k++){
            for (int j=0; j<M; j++){
                result[M * i + j] += A[K * i + k] * B[M * k + j];
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

//Multi thread
Matrix mul_mat_2(const Matrix& A, const Matrix& B){
    int N, K, M;
    N = A.get_row();
    K = A.get_col();
    M = B.get_col();

    assert(K == B.get_row());
    
    std::vector<float> result(M * N);


    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<N; i++){
        for (int k=0; k<K; k++){
            for (int j=0; j<M; j++){
                result[M * i + j] += A[K * i + k] * B[M * k + j];
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

//SIMD AVX2 256ビット　Float（32ビット）8個同時計算
Matrix mul_mat_3(const Matrix& A, const Matrix& B){
    int N, K, M;
    N = A.get_row();
    K = A.get_col();
    M = B.get_col();

    assert(K == B.get_row());
    
    std::vector<float> result(M * N);

    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<N; i++){
        for (int k=0; k<K; k++){
            __m256 a = _mm256_set1_ps(A[K * i + k]);
            
            //SIMDの処理
            for (int j=0; j<M; j+=8){
                __m256 b = _mm256_loadu_ps(&B[M * k + j]);
                __m256 res = _mm256_loadu_ps(&result[M * i + j]);

                res = _mm256_fmadd_ps(a, b, res);

                _mm256_storeu_ps(&result[M * i + j], res);
            }
            //端数処理
            for (int j=M - M%8; j<M; j++){
                result[M * i + j] += A[K * i + k] * B[M * k + j];
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

//SIMD AVX2 データパッキング
Matrix mul_mat_4(const Matrix& A, const Matrix& B){
    int N, K, M;
    N = A.get_row();
    K = A.get_col();
    M = B.get_col();

    assert(K == B.get_row());

    //データパッキング
    std::vector<float> bt(M * K, 0.0f);
    
    #pragma omp parallel for
    for (int i=0; i<K; i++){
        for (int j=0; j<M; j++){
            bt[K * j + i] = B[M * i + j];
        }
    }

    
    std::vector<float> result(M * N);

    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<N; i++){
        for (int j=0; j<M; j++){

            __m256 res = _mm256_setzero_ps();
            
            for (int k=0; k<K; k+=8){
                __m256 a = _mm256_loadu_ps(&A[K * i + k]);
                __m256 b = _mm256_loadu_ps(&bt[K * j + k]);
                
                res = _mm256_fmadd_ps(a, b, res);

                //水平加算
                __m128 lo  = _mm256_extractf128_ps(res, 0);
                __m128 hi  = _mm256_extractf128_ps(res, 1);

                __m128 v128 = _mm_add_ps(lo, hi);
    
                __m128 shuf = _mm_movehdup_ps(v128);               // 3, 1, 3, 1 の順にシャッフル
                __m128 sums = _mm_add_ps(v128, shuf);
                
                __m128 shuf2 = _mm_movehl_ps(sums, sums);          // 上位2要素を下位に移動
                __m128 final_sum = _mm_add_ps(sums, shuf2);
                float s;
                _mm_store_ss(&s, final_sum);

                result[M * i + j] += s;                  // 結果をスカラ値として取得
            }

            //端数計算
            for (int k=K - K%8; k<K; k++){
                result[M * i + j] += A[K * i + k] * bt[K * j + k];
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

// キャッシュブロッキング（タイリング） + SIMD + OpenMP
Matrix mul_mat_5(const Matrix& A, const Matrix& B){
    int N = A.get_row();
    int K = A.get_col();
    int M = B.get_col();

    assert(K == B.get_row()); 
    std::vector<float> result(M * N, 0.0f); 

    const int BLOCK_SIZE = 64; 

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < M; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                
                // ブロックがはみ出さないように安全な境界を計算
                int i_max = std::min(ii + BLOCK_SIZE, N);
                int j_max = std::min(jj + BLOCK_SIZE, M);
                int k_max = std::min(kk + BLOCK_SIZE, K);

                // 2. 内側のループ
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        __m256 vec_a = _mm256_set1_ps(A[K * i + k]);
                        
                        int j = jj;
                        // SIMDで8個ずつ処理できるところまで計算
                        for (; j <= j_max - 8; j += 8) {
                            __m256 vec_b   = _mm256_loadu_ps(&B[M * k + j]);
                            __m256 vec_res = _mm256_loadu_ps(&result[M * i + j]);
                            
                            vec_res = _mm256_fmadd_ps(vec_a, vec_b, vec_res);
                            
                            _mm256_storeu_ps(&result[M * i + j], vec_res);
                        }
                        
                        // 端数処理
                        for (; j < j_max; j++) {
                            result[M * i + j] += A[K * i + k] * B[M * k + j];
                        }
                    }
                }

            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

// SIMD AVX2 + OpenMP + ループアンローリング（4並列）
Matrix mul_mat_6(const Matrix& A, const Matrix& B){
    int N = A.get_row();
    int K = A.get_col();
    int M = B.get_col();

    assert(K == B.get_row());
    
    std::vector<float> result(M * N, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        for (int k = 0; k < K; k++){
            __m256 vec_a = _mm256_set1_ps(A[K * i + k]);
            
            int M_simd_32 = M - (M % 32);
            int j = 0;

            for (; j < M_simd_32; j += 32){
                __m256 res0 = _mm256_loadu_ps(&result[M * i + j]);
                __m256 res1 = _mm256_loadu_ps(&result[M * i + j + 8]);
                __m256 res2 = _mm256_loadu_ps(&result[M * i + j + 16]);
                __m256 res3 = _mm256_loadu_ps(&result[M * i + j + 24]);

                __m256 b0 = _mm256_loadu_ps(&B[M * k + j]);
                __m256 b1 = _mm256_loadu_ps(&B[M * k + j + 8]);
                __m256 b2 = _mm256_loadu_ps(&B[M * k + j + 16]);
                __m256 b3 = _mm256_loadu_ps(&B[M * k + j + 24]);

 
                res0 = _mm256_fmadd_ps(vec_a, b0, res0);
                res1 = _mm256_fmadd_ps(vec_a, b1, res1);
                res2 = _mm256_fmadd_ps(vec_a, b2, res2);
                res3 = _mm256_fmadd_ps(vec_a, b3, res3);

                // 計算結果を書き戻す
                _mm256_storeu_ps(&result[M * i + j], res0);
                _mm256_storeu_ps(&result[M * i + j + 8], res1);
                _mm256_storeu_ps(&result[M * i + j + 16], res2);
                _mm256_storeu_ps(&result[M * i + j + 24], res3);
            }
            
            // 2. 端数処理1
            for (; j <= M - 8; j += 8){
                __m256 b   = _mm256_loadu_ps(&B[M * k + j]);
                __m256 res = _mm256_loadu_ps(&result[M * i + j]);
                res = _mm256_fmadd_ps(vec_a, b, res);
                _mm256_storeu_ps(&result[M * i + j], res);
            }

            // 3. 端数処理2
            for (; j < M; j++){
                result[M * i + j] += A[K * i + k] * B[M * k + j];
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

// キャッシュブロッキング + ループアンローリング(4並列) + OpenMP動的スケジュール
Matrix mul_mat_7(const Matrix& A, const Matrix& B){
    int N = A.get_row();
    int K = A.get_col();
    int M = B.get_col();

    assert(K == B.get_row()); 
    std::vector<float> result(M * N, 0.0f); 

    const int BLOCK_SIZE = 64; 

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < M; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                
                int i_max = std::min(ii + BLOCK_SIZE, N);
                int j_max = std::min(jj + BLOCK_SIZE, M);
                int k_max = std::min(kk + BLOCK_SIZE, K);

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        
                        __m256 vec_a = _mm256_set1_ps(A[K * i + k]);
                        int j = jj;
                        
                        // 1. メインループ：32個（8個 × 4レジスタ）ずつ一気に処理
                        for (; j <= j_max - 32; j += 32) {
                            __m256 b0 = _mm256_loadu_ps(&B[M * k + j]);
                            __m256 b1 = _mm256_loadu_ps(&B[M * k + j + 8]);
                            __m256 b2 = _mm256_loadu_ps(&B[M * k + j + 16]);
                            __m256 b3 = _mm256_loadu_ps(&B[M * k + j + 24]);

                            __m256 res0 = _mm256_loadu_ps(&result[M * i + j]);
                            __m256 res1 = _mm256_loadu_ps(&result[M * i + j + 8]);
                            __m256 res2 = _mm256_loadu_ps(&result[M * i + j + 16]);
                            __m256 res3 = _mm256_loadu_ps(&result[M * i + j + 24]);
                            
                            // CPUのFMA演算器を並列でフル稼働させる
                            res0 = _mm256_fmadd_ps(vec_a, b0, res0);
                            res1 = _mm256_fmadd_ps(vec_a, b1, res1);
                            res2 = _mm256_fmadd_ps(vec_a, b2, res2);
                            res3 = _mm256_fmadd_ps(vec_a, b3, res3);
                            
                            _mm256_storeu_ps(&result[M * i + j], res0);
                            _mm256_storeu_ps(&result[M * i + j + 8], res1);
                            _mm256_storeu_ps(&result[M * i + j + 16], res2);
                            _mm256_storeu_ps(&result[M * i + j + 24], res3);
                        }

                        // 2. 端数処理
                        // 8個ずつ処理
                        for (; j <= j_max - 8; j += 8) {
                            __m256 b   = _mm256_loadu_ps(&B[M * k + j]);
                            __m256 res = _mm256_loadu_ps(&result[M * i + j]);
                            res = _mm256_fmadd_ps(vec_a, b, res);
                            _mm256_storeu_ps(&result[M * i + j], res);
                        }
                        
                        // 1個ずつ処理
                        for (; j < j_max; j++) {
                            result[M * i + j] += A[K * i + k] * B[M * k + j];
                        }
                    }
                }
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}

Matrix mul_mat_8(const Matrix& A, const Matrix& B){
    int N = A.get_row();
    int K = A.get_col();
    int M = B.get_col();

    assert(K == B.get_row()); 
    std::vector<float> result(M * N, 0.0f); 

    // L2キャッシュサイズとTLBの限界から逆算したブロックサイズ
    const int BLOCK_SIZE = 128; 

    // スレッドの起動オーバーヘッドを減らすため、外側で parallel ブロックを作る
    #pragma omp parallel
    {
        // 【魔法のバッファ】
        // 各スレッド専用の、32バイト境界に揃えられた「完全連続メモリ」
        // これにより、TLBミスが完全に消滅し、プリフェッチャが限界突破する
        alignas(32) float pack_B[BLOCK_SIZE * BLOCK_SIZE];

        #pragma omp for schedule(dynamic)
        for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < M; jj += BLOCK_SIZE) {
                for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                    
                    int i_max = std::min(ii + BLOCK_SIZE, N);
                    int j_max = std::min(jj + BLOCK_SIZE, M);
                    int k_max = std::min(kk + BLOCK_SIZE, K);

                    int b_width = j_max - jj;
                    int b_height = k_max - kk;

                    // ----------------------------------------------------
                    // 1. パッキングフェーズ (Packing)
                    // ----------------------------------------------------
                    // 行列Bの「飛び飛び」になっている該当ブロックだけを、
                    // 計算前に alignas(32) の連続したバッファ(pack_B)に詰め替える！
                    for (int k = 0; k < b_height; k++) {
                        // 高速なメモリコピーを使用
                        std::memcpy(&pack_B[k * b_width], &B[M * (kk + k) + jj], b_width * sizeof(float));
                    }

                    // ----------------------------------------------------
                    // 2. マイクロカーネル計算フェーズ (Micro-kernel)
                    // ----------------------------------------------------
                    for (int i = ii; i < i_max; i++) {
                        for (int k = 0; k < b_height; k++) {
                            
                            __m256 vec_a = _mm256_set1_ps(A[K * i + (kk + k)]);
                            int j = 0;
                            
                            // メインループ：32個並列処理
                            // ここで読み込んでいるのは B ではなく pack_B（完全連続データ）
                            for (; j <= b_width - 32; j += 32) {
                                __m256 b0 = _mm256_loadu_ps(&pack_B[k * b_width + j]);
                                __m256 b1 = _mm256_loadu_ps(&pack_B[k * b_width + j + 8]);
                                __m256 b2 = _mm256_loadu_ps(&pack_B[k * b_width + j + 16]);
                                __m256 b3 = _mm256_loadu_ps(&pack_B[k * b_width + j + 24]);

                                __m256 res0 = _mm256_loadu_ps(&result[M * i + jj + j]);
                                __m256 res1 = _mm256_loadu_ps(&result[M * i + jj + j + 8]);
                                __m256 res2 = _mm256_loadu_ps(&result[M * i + jj + j + 16]);
                                __m256 res3 = _mm256_loadu_ps(&result[M * i + jj + j + 24]);
                                
                                res0 = _mm256_fmadd_ps(vec_a, b0, res0);
                                res1 = _mm256_fmadd_ps(vec_a, b1, res1);
                                res2 = _mm256_fmadd_ps(vec_a, b2, res2);
                                res3 = _mm256_fmadd_ps(vec_a, b3, res3);
                                
                                _mm256_storeu_ps(&result[M * i + jj + j], res0);
                                _mm256_storeu_ps(&result[M * i + jj + j + 8], res1);
                                _mm256_storeu_ps(&result[M * i + jj + j + 16], res2);
                                _mm256_storeu_ps(&result[M * i + jj + j + 24], res3);
                            }

                            // 端数処理（8個ずつ）
                            for (; j <= b_width - 8; j += 8) {
                                __m256 b   = _mm256_loadu_ps(&pack_B[k * b_width + j]);
                                __m256 res = _mm256_loadu_ps(&result[M * i + jj + j]);
                                res = _mm256_fmadd_ps(vec_a, b, res);
                                _mm256_storeu_ps(&result[M * i + jj + j], res);
                            }
                            
                            // 端数処理（1個ずつ）
                            for (; j < b_width; j++) {
                                result[M * i + jj + j] += A[K * i + (kk + k)] * pack_B[k * b_width + j];
                            }
                        }
                    }
                }
            }
        }
    }

    Matrix C(N, M);
    C.set_data(result);

    return C;
}


//測定用関数
void measure_mat(const Matrix& A, const Matrix& B, const std::string& function_name, const std::function<Matrix(Matrix, Matrix)>& callback){
    auto start = std::chrono::system_clock::now();
    Matrix C = callback(A, B);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("%s %8.2fms\n", function_name.c_str(), elapsed.count());
}

#define MEASURE_MAT(A, B, func) measure_mat(A, B, #func, func)

std::vector<float> generate_vector(int N) {
    std::mt19937 mt( std::random_device{}() );
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    std::vector<float> result(N);
    std::generate(result.begin(), result.end(), [&dist, &mt](){ return dist(mt) ;});
    return result;
}

int main(){
    
    const int N = 4096;
    const int K = 4096;
    const int M = 4096;

    Matrix A(N, K);
    Matrix B(K, M);
    
    std::vector<float> v = generate_vector(N * K);
    std::vector<float> e = generate_vector(K * M);
    
    //初期化
    A.set_data(v);
    B.set_data(e);
    
    // MEASURE_MAT(A, B, mul_mat_0);
    MEASURE_MAT(A, B, mul_mat_1);
    MEASURE_MAT(A, B, mul_mat_2);
    MEASURE_MAT(A, B, mul_mat_3);
    MEASURE_MAT(A, B, mul_mat_4);
    MEASURE_MAT(A, B, mul_mat_5);
    MEASURE_MAT(A, B, mul_mat_6);
    MEASURE_MAT(A, B, mul_mat_7);
    MEASURE_MAT(A, B, mul_mat_8);
    return 0;
}