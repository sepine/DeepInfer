if [ -z "$1" ]; then
    echo 'Please specify a test batch size!'
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=num --test_only=True --test_batch_size=$1 --language=compiler0.5 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=1 --test_only=True --test_batch_size=$1 --language=compiler0.5 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=2 --test_only=True --test_batch_size=$1 --language=compiler0.5 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=3 --test_only=True --test_batch_size=$1 --language=compiler0.5 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=4 --test_only=True --test_batch_size=$1 --language=compiler0.5 --batch_size=64 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=5 --test_only=True --test_batch_size=$1 --language=compiler0.5 --batch_size=64 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=6 --test_only=True --test_batch_size=$1 --language=compiler0.5 --batch_size=32 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=7 --test_only=True --test_batch_size=$1 --language=compiler0.5 --batch_size=32 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=8 --test_only=True --test_batch_size=$1 --language=compiler0.5 --batch_size=16 --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_return.py --pos=9 --test_only=True --test_batch_size=$1 --language=compiler0.5 --batch_size=16 --epochs=50;
sleep 1;

python run_return.py --print_all=True --language=compiler0.5


rm -rf ./results/compiler0.5/return/results.pkl