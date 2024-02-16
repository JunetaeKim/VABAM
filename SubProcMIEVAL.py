import subprocess
import time
import argparse

# Refer to the execution code
# python .\SubProcMIEVAL.py --Config EvalConfigART500 --ConfigSpec TCMIDKZFC_ART_30_500  --GPUID 4 --SpecNZs 10 20
def run_script(Config, ConfigSpec, SpecNZs, GPUID):
    ConfigArg = f"--Config={Config}"
    GPUIDArg = f"--GPUID={GPUID}"

    # When SpecNZs is not None
    if SpecNZs is not None:
        for NZs in SpecNZs:
            command = ["python", 'BatchMIEvaluation.py', ConfigArg, GPUIDArg]

            # Include ConfigSpec arguments if they are not None
            if ConfigSpec is not None:
                ConfigSpecArgs = [f"--ConfigSpec={spec}" for spec in ConfigSpec]
                command += ConfigSpecArgs

            ArgNZs = f"--SpecNZs={NZs}"
            command.append(ArgNZs)

            subprocess.run(command)
            time.sleep(1)
    # When SpecNZs is None
    else:
        command = ["python", 'BatchMIEvaluation.py', ConfigArg, GPUIDArg]

        # Include ConfigSpec arguments if they are not None
        if ConfigSpec is not None:
            ConfigSpecArgs = [f"--ConfigSpec={spec}" for spec in ConfigSpec]
            command += ConfigSpecArgs

        subprocess.run(command)
        time.sleep(1)


# Parse arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the YAML file).')
parser.add_argument('--ConfigSpec', nargs='+', type=str, required=False, 
                    default=None, help='Set the name of the specific configuration to load (the name of the model config in the YAML file).')
parser.add_argument('--SpecNZs', nargs='+', type=int, required=False, 
                    default=None, help='Set the size of js to be selected at the same time with the list.')
parser.add_argument('--GPUID', type=int, required=False, default=1)

args = parser.parse_args()

# Execute the script
run_script(args.Config, args.ConfigSpec, args.SpecNZs, args.GPUID)
