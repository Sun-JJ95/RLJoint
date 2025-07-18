import itertools
import subprocess
import re
import os

# 定义参数组合
batch_sizes = [32]
learning_rates = [0.1]
epochs = [300]
decay_rates = [0.0001]
npoints = [8192]
lr_decays = [0.5]
checkpoint_file = 'completed_combinations.txt'


# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


# 写入文件内容
def write_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)


# 记录已完成的组合
def log_completed_combination(combo):
    with open(checkpoint_file, 'a') as file:
        file.write(f"{combo}\n")


# 读取已完成的组合
def read_completed_combinations():
    if not os.path.exists(checkpoint_file):
        return set()
    with open(checkpoint_file, 'r') as file:
        return set(line.strip() for line in file)


# 修改参数并运行脚本
def modify_and_run(file_path):
    original_lines = read_file(file_path)
    completed_combinations = read_completed_combinations()

    for batch_size, learning_rate, epoch, decay_rate, npoint, lr_decay in itertools.product(
            batch_sizes, learning_rates, epochs, decay_rates, npoints, lr_decays):

        combo = f"{batch_size},{learning_rate},{epoch},{decay_rate},{npoint},{lr_decay}"
        if combo in completed_combinations:
            continue

        new_lines = []
        for line in original_lines:
            if "--batch_size" in line:
                line = re.sub(r"default=\d+", f"default={batch_size}", line)
            elif "--learning_rate" in line:
                line = re.sub(r"default=[\d.]+", f"default={learning_rate}", line)
            elif "--epoch" in line:
                line = re.sub(r"default=\d+", f"default={epoch}", line)
            elif "--decay_rate" in line:
                line = re.sub(r"default=[\d.]+", f"default={decay_rate}", line)
            elif "--npoint" in line:
                line = re.sub(r"default=\d+", f"default={npoint}", line)
            elif "--lr_decay" in line:
                line = re.sub(r"default=[\d.]+", f"default={lr_decay}", line)
            new_lines.append(line)

        # 输出当前参数组合
        print(
            f"Running with batch_size={batch_size}, learning_rate={learning_rate}, epoch={epoch}, decay_rate={decay_rate}, npoint={npoint}, lr_decay={lr_decay}")

        # 写入修改后的内容
        write_file(file_path, new_lines)

        # 运行脚本
        try:
            subprocess.run(["/root/miniconda3/bin/python3", file_path], check=True)
            log_completed_combination(combo)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")


# 调用函数
modify_and_run('MyTrainFun.py')


