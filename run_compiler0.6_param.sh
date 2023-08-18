if [ -z "$1" ]; then
    echo 'Please specify a test batch size!'
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=num --test_only=True --test_batch_size=$1 --language=compiler0.6 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=1 --test_only=True --test_batch_size=$1 --language=compiler0.6 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=2 --test_only=True --test_batch_size=$1 --language=compiler0.6 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=3 --test_only=True --test_batch_size=$1 --language=compiler0.6 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=4 --test_only=True --test_batch_size=$1 --language=compiler0.6 --batch_size=64 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=5 --test_only=True --test_batch_size=$1 --language=compiler0.6 --batch_size=64 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=6 --test_only=True --test_batch_size=$1 --language=compiler0.6 --batch_size=64 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=7 --test_only=True --test_batch_size=$1 --language=compiler0.6 --batch_size=32 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=8 --test_only=True --test_batch_size=$1 --language=compiler0.6 --batch_size=32 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=9 --test_only=True --test_batch_size=$1 --language=compiler0.6 --batch_size=16 --model=classify --module=param --epochs=50;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=1 --test_only=True --test_batch_size=$1 --learning_rate=1e-3 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=2 --test_only=True --test_batch_size=$1 --learning_rate=5e-4 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=3 --test_only=True --test_batch_size=$1 --learning_rate=5e-4 --batch_size=64 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=4 --test_only=True --test_batch_size=$1 --learning_rate=5e-4 --batch_size=64 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=5 --test_only=True --test_batch_size=$1 --learning_rate=5e-4 --batch_size=32 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=6 --test_only=True --test_batch_size=$1 --batch_size=32 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=7 --test_only=True --test_batch_size=$1 --batch_size=16 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=8 --test_only=True --test_batch_size=$1 --batch_size=16 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

CUDA_VISIBLE_DEVICES=0 python run_param.py --pos=9 --test_only=True --test_batch_size=$1 --batch_size=16 --language=compiler0.6 --model=gene --module=param --epochs=30;
sleep 1;

python run_param.py --print_all=True --language=compiler0.6 --module=param


rm -rf ./results/compiler0.6/param/results.pkl