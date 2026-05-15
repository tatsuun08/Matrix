#pragma once
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

class Matrix {
    private:
        int rows_;
        int cols_;
        std::vector<float> data_;
    public:
        //コンストラクタ
        Matrix(int rows, int cols) :  cols_(cols), rows_(rows) {
            data_.resize(rows_ * cols_);
        }

        int size(){ return rows_ * cols_ ;}

        float operator[](int n) const {
            assert(0 <= n && n < size());
            return data_[n] ;
        }

        float& operator[](int n){
            assert(0 <= n && n < size());
            return data_[n] ;
        }
        

        //ゲッター
        int get_col() const { return cols_ ;}
        int get_row() const { return rows_ ;}

        //セッター
        void set_data(const std::vector<float>& input){
            if (input.size() != size()){
                printf("行列の要素数が合いません.　期待している要素数 = %d　入力された要素数 = %d", size(), input.size());
                std::exit(EXIT_FAILURE);
            }
            else {
                data_ = input;
            }
        }

        //行列の表示
        void print(){
            for (int i=0; i<rows_; i++){
                for (int j=0; j<cols_; j++){
                    std::cout << data_[cols_ * i + j];

                    if (j != cols_-1) std::cout << " ";
                    else std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }
};