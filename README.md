# Supervoxel merging pipeline

Semi-automatic segmentation workflow for Volume SEM datasets

## Installation 

Paintera:

    conda env create -f paintera-env.yml
    

Main environment:

    conda env create -f paintera-mc-workflow-env.yml

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