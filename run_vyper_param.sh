if [ -z "$1" ]; then
    echo 'Please specify a test batch size!'
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=num --test_only=True --test_batch_size=$1 --language=vyper --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=1 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=64 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=2 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=64 --learning_rate=5e-4 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=3 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=32 --learning_rate=5e-4 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=4 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=32 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=5 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=16 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=6 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=16 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=7 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=8 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=8 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=8 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=9 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=8 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=1 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=32 --learning_rate=5e-4 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=2 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=16 --learning_rate=5e-4 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=3 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=8 --learning_rate=5e-4 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=4 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=4 --learning_rate=5e-4 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=5 --test_only=True --test_batch_size=$1 --language=vyper --batch_size=4 --learning_rate=5e-4 --model=gene --module=param --epochs=30;
sleep 1;

python run_param.py --print_all=True --language=vyper --module=param


rm -rf ./results/vyper/param/results.pkl