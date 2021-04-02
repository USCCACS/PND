#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30
#SBATCH --output="PND-sbatch-test-%A.out"

tar -cvzf PND.tgz PND/

[ -f "build_run_PND_Discovery.sh" ]  && rm build_run_PND_Discovery.sh

cp PND/build_run_PND_Discovery.sh .

sh build_run_PND_Discovery.sh