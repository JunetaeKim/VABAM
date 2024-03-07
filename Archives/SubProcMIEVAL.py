import subprocess
import time
import argparse
from itertools import product

# Refer to the execution code
# python .\SubProcMIEVAL.py --Config EvalConfigART500 --ConfigSpec TCMIDKZFC_ART_30_500  --GPUID 4 --SpecNZs 10 20
# python .\SubProcMIEVAL.py --Config EvalConfigII800 --GPUID 4 --SpecNZs  20 --SpecFCs 0.1  0.8
def run_command_with_args(Config, ConfigSpec, GPUID, SpecNZs=None, SpecFCs=None):
    ConfigArg = f"--Config={Config}"
    GPUIDArg = f"--GPUID={GPUID}"
    command_base = ["python", 'BatchMIEvaluation.py', ConfigArg, GPUIDArg]
    
    # Include ConfigSpec arguments if they are not None
    if ConfigSpec is not None:
        ConfigSpecArgs = [f"--ConfigSpec={spec}" for spec in ConfigSpec]
        command_base += ConfigSpecArgs
    
    # Default to [''] if None to ensure product works correctly
    SpecNZs = SpecNZs if SpecNZs is not None else ['']
    SpecFCs = SpecFCs if SpecFCs is not None else ['']
    
    # Use itertools.product to prepare combinations
    combinations = product(SpecNZs, SpecFCs)

    for NZs, FCs in combinations:
        command_specific = command_base.copy()
        if NZs: command_specific.append(f"--SpecNZs={NZs}")
        if FCs: command_specific.append(f"--SpecFCs={FCs}")
        print(command_specific)
        subprocess.run(command_specific)
        time.sleep(1)

def run_script(Config, ConfigSpec, GPUID, SpecNZs, SpecFCs):
    run_command_with_args(Config, ConfigSpec, GPUID, SpecNZs, SpecFCs)


# Parse arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Config', type=str, required=True, help='Set the name of the configuration to load (the name of the YAML file).')
parser.add_argument('--ConfigSpec', nargs='+', type=str, required=False, 
                    default=None, help='Set the name of the specific configuration to load (the name of the model config in the YAML file).')
parser.add_argument('--SpecNZs', nargs='+', type=int, required=False, 
                    default=None, help='Set the size of js to be selected at the same time with the list.')
parser.add_argument('--SpecFCs', nargs='+', type=float, required=False, default=None,
                    help='Set the frequency cutoff range(s) for signal synthesis. Multiple ranges can be provided.')
parser.add_argument('--GPUID', type=int, required=False, default=1)

args = parser.parse_args()

# Execute the script
run_script(args.Config, args.ConfigSpec, args.GPUID, args.SpecNZs, args.SpecFCs)
