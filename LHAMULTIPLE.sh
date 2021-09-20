#!/bin/bash


#To tile extraction:
bash LHA.sh tile ./configure/lha/tile_ds1.json
bash LHA.sh tile ./configure/lha/tile_ds2.json
bash LHA.sh tile ./configure/lha/tile_ds3.json

#To feature extraction:
bash LHA.sh description ./configure/lha/desc_ds1_LBP.json
bash LHA.sh description ./configure/lha/desc_ds2_LBP.json
bash LHA.sh description ./configure/lha/desc_ds3_LBP.json

bash LHA.sh description ./configure/lha/desc_ds1_RAD.json
bash LHA.sh description ./configure/lha/desc_ds2_RAD.json
bash LHA.sh description ./configure/lha/desc_ds3_RAD.json

#To classification:
bash LHA.sh classification ./configure/lha/class_ds1_LBP.json
bash LHA.sh classification ./configure/lha/class_ds2_LBP.json
bash LHA.sh classification ./configure/lha/class_ds3_LBP.json

bash LHA.sh classification ./configure/lha/class_ds1_RAD.json
bash LHA.sh classification ./configure/lha/class_ds2_RAD.json
bash LHA.sh classification ./configure/lha/class_ds3_RAD.json

bash LHA.sh classification ./configure/lha/class_ds1_LBP+RAD.json
bash LHA.sh classification ./configure/lha/class_ds2_LBP+RAD.json
bash LHA.sh classification ./configure/lha/class_ds3_LBP+RAD.json


#paper 
bash LHA.sh classification ./configure/lha/class_ds3_LBP_filter.json


########### GRIDSEARCH
bash LHA.sh classification ./configure/lha/class_ds3_GRID_LBP_filter.json
bash LHA.sh classification ./configure/lha/class_ds3_GRID_RAD_filter.json
bash LHA.sh classification ./configure/lha/class_ds3_GRID_LBP+RAD_filter.json

########### feature selection
bash LHA.sh classification ./configure/lha/class_ds3_FESE_LBP+RAD_SFF.json 

########### CROSS VALIDATION K=10
bash LHA.sh classification ./configure/lha/class_ds3_CROSS_RAD_filter
bash LHA.sh classification ./configure/lha/class_ds3_CROSS_LBP_filter
bash LHA.sh classification ./configure/lha/class_ds3_CROSS_LBP+RAD_filter

########### SIMPLE CLASSIFICATION
bash LHA.sh classification ./configure/lha/class_ds3_SIMP_LBP_filter.json
bash LHA.sh classification ./configure/lha/class_ds3_SIMP_RAD_filter.json
bash LHA.sh classification ./configure/lha/class_ds3_SIMP_LBP+RAD_filter.json
