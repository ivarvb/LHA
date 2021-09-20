## LHA
>Lung Histopathological Analysis (LHA)

## To install:
>* bash LHA.sh install. 

## To execute:
>#### To tile extraction:
>>* bash LHA.sh tile ./configure/lha/tile_ds1.json
>>* bash LHA.sh tile ./configure/lha/tile_ds2.json
>>* bash LHA.sh tile ./configure/lha/tile_ds3.json
>#### To feature extraction:
>>* bash LHA.sh description ./configure/lha/desc_ds1_LBP.json
>>* bash LHA.sh description ./configure/lha/desc_ds1_RAD.json
>----
>>* bash LHA.sh description ./configure/lha/desc_ds2_LBP.json
>>* bash LHA.sh description ./configure/lha/desc_ds2_RAD.json
>>----
>>* bash LHA.sh description ./configure/lha/desc_ds3_LBP.json
>>* bash LHA.sh description ./configure/lha/desc_ds3_RAD.json

>#### To join two descriptors (LBP+RAD):
>>* bash LHA.sh join ./configure/lha/join_ds1_LBP+RAD.json
>>* bash LHA.sh join ./configure/lha/join_ds2_LBP+RAD.json
>>* bash LHA.sh join ./configure/lha/join_ds3_LBP+RAD.json

>#### To classification:
>>* bash LHA.sh classification ./configure/lha/class_ds1_LBP.json
>>* bash LHA.sh classification ./configure/lha/class_ds1_RAD.json
>>* bash LHA.sh classification ./configure/lha/class_ds1_LBP+RAD.json
>>----
>>* bash LHA.sh classification ./configure/lha/class_ds2_LBP.json
>>* bash LHA.sh classification ./configure/lha/class_ds2_RAD.json
>>* bash LHA.sh classification ./configure/lha/class_ds2_LBP+RAD.json
>>----
>>* bash LHA.sh classification ./configure/lha/class_ds3_LBP.json
>>* bash LHA.sh classification ./configure/lha/class_ds3_RAD.json
>>* bash LHA.sh classification ./configure/lha/class_ds3_LBP+RAD.json

>### To visualize:
>>* bash LHA.sh visualization ./configure/lha/vis_ds3.json

## To configure the parameters:
>#### To tile extraction:
>>* edit: ./configure/lha/tile_ds1.json
>>* edit: ./configure/lha/tile_ds2.json
>>* edit: ./configure/lha/tile_ds3.json
>#### To feature extraction:
>>* edit: ./configure/lha/desc_ds1_LBP.json
>>* edit: ./configure/lha/desc_ds1_RAD.json
>>* edit: ./configure/lha/desc_ds1_LBP+RADV1.json
>>----
>>* edit: ./configure/lha/desc_ds2_LBP.json
>>* edit: ./configure/lha/desc_ds2_RAD.json
>>* edit: ./configure/lha/desc_ds2_LBP+RADV1.json
>>----
>>* edit: ./configure/lha/desc_ds3_LBP.json
>>* edit: ./configure/lha/desc_ds3_RAD.json
>>* edit: ./configure/lha/desc_ds3_LBP+RADV1.json
>#### To join two descriptors (LBP+RAD):
>>* edit: ./configure/lha/join_ds1_LBP+RAD.json
>>* edit: ./configure/lha/join_ds2_LBP+RAD.json
>>* edit: ./configure/lha/join_ds3_LBP+RAD.json
>#### To classification:
>>* edit: ./configure/lha/class_ds1_LBP.json
>>* edit: ./configure/lha/class_ds1_RAD.json
>>* edit: ./configure/lha/class_ds1_LBP+RAD.json
>>----
>>* edit: ./configure/lha/class_ds2_LBP.json
>>* edit: ./configure/lha/class_ds2_RAD.json
>>* edit: ./configure/lha/class_ds2_LBP+RAD.json
>>----
>>* edit: ./configure/lha/class_ds3_LBP.json
>>* edit: ./configure/lha/class_ds3_RAD.json
>>* edit: ./configure/lha/class_ds3_LBP+RAD.json
>#### To visualize:
>>* edit: ./configure/lha/vis_ds3.json




## Pipeline:
>* bash LHA.sh tile ./configure/lha/tile_ds3.json
>* bash LHA.sh description ./configure/lha/desc_ds3_LBP_V2.json
>* bash LHA.sh description ./configure/lha/desc_ds3_RAD_V2.json

>* bash LHA.sh join ./configure/lha/join_ds3_LBP+RAD_V2.json

>* bash LHA.sh classification ./configure/lha/class_ds3_V2.json

>* bash LHA.sh join ./configure/lha/join_ds3_LBP+RAD_V2_predicted.json



