# Supervoxel merging pipeline

Semi-automatic segmentation workflow for Volume SEM datasets

## Installation 

### From environment files

Paintera:

    conda env create -f paintera-env.yml
    
Main environment:

    conda env create -f paintera-mc-workflow-env.yml
    
### Manually

Paintera

    TODO
    
Main environment:

    conda create -n paintera-mc-workflow-env -c cpape -c conda-forge elf python=3.7
    conda activate paintera-mc-workflow-env
    conda install -c conda-forge napari
    conda install -c cpape z5py
    
## Usage

    from paintera_multicut_workflow import pm_workflow

    pm_workflow(
        results_folder='path/to/results/folder/',
        raw_filepath='path/to/raw_data.h5,
        mem_pred_filepath='path/to/mem_pred.h5',
        supervoxel_filepath='path/to/supervoxels.h5',
        mem_pred_channel=2,  # Required if mem_pred.ndim == 4 to select channel
        auto_crop_center=True,  # Crops to the center if raw, mem or sv shapes are > annotation_shape
        annotation_shape=(256, 256, 256),
        paintera_env_name='paintera_env_new',  # Name of the paintera environment
        activation_command='source /home/hennies/miniconda3/bin/activate', 
        export_binary=True,
        conncomp_on_paintera_export=True,  
        verbose=True
    )

## Installation and usage from archive (under development)

### Linux

Download pm_workflow.tar.gz

Unpack with

    mkdir pm_workflow
    tar -xzf pm_workflow.tar.gz -C pm_workflow
    
Run the pipeline

    cd pm_workflow
    ./run_workflow.sh input_folder result_folder [arguments]
    
Use help for description of arguments

    ./run_workflow.sh -h
    
### Windows

Download pm_workflow_win.zip

Unpack to the the disired location (workflow directory)

Run the pipeline using the command prompt:

Open command prompt and navigate to the workflow directory, e.g.

    cd path\to\pm_workflow_win
    
Run the pipeline with

    run_workflow.bat input_folder result_folder [arguments]
    
Use help for description of arguments

    run_workflow.bat -h

    
