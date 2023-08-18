if [ -z "$1" ]; then
    echo 'Please specify a test batch size!'
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=num --test_only=True --test_batch_size=$1 --language=vyper --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=1 --test_only=True --test_batch_size=$1 --language=vyper --learning_rate=1e-3 --epochs=200;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=2 --batch_size=32 --test_only=True --test_batch_size=$1 --language=vyper --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=3 --batch_size=32 --test_only=True --test_batch_size=$1 --language=vyper --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=4 --batch_size=16 --test_only=True --test_batch_size=$1 --language=vyper --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=5 --batch_size=16 --test_only=True --test_batch_size=$1 --language=vyper --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=6 --batch_size=8 --test_only=True --test_batch_size=$1 --language=vyper --epochs=50;
sleep 1;

python run_return.py --print_all=True --language=vyper


rm -rf ./results/vyper/return/results.pkl