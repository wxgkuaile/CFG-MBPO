import subprocess

def run_shell_script(env_name, num_e):
    # 设置shell命令
    shell_command = f'sh run.sh {env_name} {num_e}'
    
    # 调用shell命令
    subprocess.run(shell_command, shell=True)

# 在这里调用函数，并传入环境名和epoch数
env_name = 'AntTruncatedObsEnv'
num_e = 300
run_shell_script(env_name, num_e)
