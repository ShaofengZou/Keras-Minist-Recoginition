#!/bin/bash
nohup python3 main.py -gi 0 -id Exp_Base > log/Base.out &
nohup python3 main.py -gi 1 -id Exp_BlockNum1 -bn 1 > log/Exp_BlockNum1.out &
nohup python3 main.py -gi 2 -id Exp_BlockNum3 -bn 3 > log/Exp_BlockNum3.out &
nohup python3 main.py -gi 3 -id Exp_FilterNum16 -fn 16 > log/Exp_Exp_FilterNum16.out &
nohup python3 main.py -gi 4 -id Exp_FilterNum64 -fn 64 > log/Exp_Exp_FilterNum64.out &
nohup python3 main.py -gi 5 -id Exp_KernelSize3 -ks 3 > log/Exp_KernelSize3.out &
nohup python3 main.py -gi 6 -id Exp_KernelSize7 -ks 7 > log/Exp_KernelSize7.out &
nohup python3 main.py -gi 7 -id Exp_RandomNniform -i random_uniform > log/Exp_RandomNniform.out &


# nohup python3 main.py -gi 0 -id Exp_BatchNormalization -abn > log/Exp_BatchNormalization.out &
# nohup python3 main.py -gi 1 -id Exp_Orthogonal -i orthogonal > log/Exp_Orthogonal.out &