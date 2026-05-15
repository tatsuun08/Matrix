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
Matrix mul_mat(const Matrix& A, const Matrix& B){
    int N, K, M;
    N = A.get_row();
    K = A.get_col();
    M = B.get_col();

    assert(K == B.get_height());
    
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

    assert(K == B.get_height());
    
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

    assert(K == B.get_height());
    
    std::vector<float> result(M * N);


    #pragma omp parallel for
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
Matrix mul_mat_2(const Matrix& A, const Matrix& B){
    int N, K, M;
    N = A.get_row();
    K = A.get_col();
    M = B.get_col();

    assert(K == B.get_height());
    
    std::vector<float> result(M * N);

    #pragma omp parallel for
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

//測定用関数
void measure_mat(const Matrix& A, const Matrix& B, const std::string& function_name, const std::function<Matrix(Matrix, Matrix)>& callback){
    auto start = std::chrono::system_clock::now();
    Matrix C = callback(A, B);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << function_name << " ";
    std::cout << elapsed.count() << "ms" << std::endl;

    // C.print();
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
    
    const int N = 1024;
    const int K = 1024;
    const int M = 1024;

    Matrix A(N, K);
    Matrix B(K, M);
    
    std::vector<float> v = generate_vector(N * K);
    std::vector<float> e = generate_vector(K * M);
    
    //初期化
    A.set_data(v);
    B.set_data(e);
    
    MEASURE_MAT(A, B, mul_mat);
    MEASURE_MAT(A, B, mul_mat_1);
    MEASURE_MAT(A, B, mul_mat_2);

    return 0;
}