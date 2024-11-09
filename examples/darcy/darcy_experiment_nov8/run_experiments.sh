(
    for seed in 12 43 530 13; do
        for num_functions in 2 5 10 20 30 40 50; do
            echo "Running with num_functions=$num_functions , seed=$seed on device_num=1"
            python run_darcy_experiment.py \
                --device_num 1 \
                --num_functions $num_functions \
                --samples_per_function 5 \
                --collocation_grid_n 16 \
                --a_matern_lengthscale 0.25 \
                --a_exponent 0.25 \
                --random_seed $seed \
                --experiment_extra_name "main_run"
        done
    done
) &
(
    for seed in 12 43 530 13; do
        for num_functions in 2 5 10 20 30 40 50; do
            echo "Running with num_functions=$num_functions , seed=$seed on device_num=2"
            python run_darcy_experiment.py \
                --device_num 2 \
                --num_functions $num_functions \
                --samples_per_function 10 \
                --collocation_grid_n 16 \
                --a_matern_lengthscale 0.25 \
                --a_exponent 0.25 \
                --random_seed $seed \
                --experiment_extra_name "main_run"
        done
    done
) &
(
    for seed in 12 43 530 13; do
        for num_functions in 2 5 10 20 30 40 50; do
            echo "Running with num_functions=$num_functions , seed=$seed on device_num=3"
            python run_darcy_experiment.py \
                --device_num 3 \
                --num_functions $num_functions \
                --samples_per_function 20 \
                --collocation_grid_n 16 \
                --a_matern_lengthscale 0.25 \
                --a_exponent 0.25 \
                --random_seed $seed \
                --experiment_extra_name "main_run"
        done
    done
) & 
