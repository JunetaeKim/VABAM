import subprocess
import time
import argparse



def run_script(Config, ConfigSpec, SpecNZs, GPUID):
    ConfigArg = f"--Config={Config}"
    ConfigSpecArgs = [f"--ConfigSpec={spec}" for spec in ConfigSpec]
    GPUIDArg = f"--GPUID={GPUID}"  # 올바른 인자 이름을 사용합니다.

    for NZs in SpecNZs:
        ArgNZs = f"--SpecNZs={NZs}"
        subprocess.run(["python", 'BatchMIEvaluation.py', ConfigArg, GPUIDArg] + ConfigSpecArgs + [ArgNZs])
        time.sleep(1)

# argparse를 사용하여 인자를 파싱합니다.
parser = argparse.ArgumentParser()
parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the YAML file).')
parser.add_argument('--ConfigSpec', nargs='+', type=str, required=False, 
                    default=None, help='Set the name of the specific configuration to load (the name of the model config in the YAML file).')
parser.add_argument('--SpecNZs', nargs='+', type=int, required=False, 
                    default=None, help='Set the size of js to be selected at the same time with the list.')
parser.add_argument('--GPUID', type=int, required=False, default=1)

args = parser.parse_args()

# 스크립트 실행
run_script(args.Config, args.ConfigSpec, args.SpecNZs, args.GPUID)
