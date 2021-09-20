#!/bin/bash

ev="./python/python38lha/"
pathlha="./sourcecode/src/vx/lha/"
install () {
    #sudo apt-get install python3-venv
    #sudo apt install python3-pip
    #sudo apt-get install libsuitesparse-dev
    #sudo apt install libx11-dev
    #############sudo apt install nvidia-cuda-toolkit


    rm -r $ev
    mkdir $ev
    python3 -m venv $ev
    source $ev"bin/activate"

    # install packages
    # pip3 install -r requirements.txt
    pip3 install wheel
    pip3 install numpy
    pip3 install scikit-sparse
    pip3 install matplotlib
    pip3 install pandas
    pip3 install opencv-python
    pip3 install scikit-image
    pip3 install -U scikit-learn
    pip3 install xgboost
    pip3 install ujson
    pip3 install seaborn
    pip3 install cython
    pip3 install xgboost

    pip3 install SimpleITK
    pip3 install pyradiomics
    pip3 install thundersvm-cpu
    pip3 install sklearn-genetic
    pip3 install mlxtend
    pip3 install imblearn

    #compile
}
#compile () {
#    source $ev"bin/activate"
#    cd ./sourcecode/src/vx/com/px/image/
#    sh Makefile.sh
#    
#}
#execute () {
#    source $ev"bin/activate"
#    cd ./sourcecode/src/vx/lha/
#    python3 Main.py
#}

visualization () {
    source $ev"bin/activate"
    file=$(pwd)"/"$FILEINPUT
    cd $pathlha
    python3 Visualization.py $file
}
classification () {
    source $ev"bin/activate"
    file=$(pwd)"/"$FILEINPUT
    cd $pathlha
    python3 Classification.py $file
}
description () {
    source $ev"bin/activate"
    file=$(pwd)"/"$FILEINPUT
    cd $pathlha
    python3 Description.py $file

}
tile () {
    source $ev"bin/activate"
    file=$(pwd)"/"$FILEINPUT
    cd $pathlha
    python3 ROI.py $file
}
join () {
    source $ev"bin/activate"
    file=$(pwd)"/"$FILEINPUT
    cd $pathlha
    python3 Join.py $file
}
plot () {
    source $ev"bin/activate"
    file=$(pwd)"/"$FILEINPUT
    cd $pathlha
    python3 Plot.py $file
}
metric () {
    source $ev"bin/activate"
    file=$(pwd)"/"$FILEINPUT
    cd $pathlha
    python3 Metric.py $file
}


args=("$@")
T1=${args[0]}
FILEINPUT=${args[1]}
#fileconten=$(<"$2")
#echo "DSD:"$fileconten
#echo "DSD1111"
#echo $FILEINPUT
if [ "$T1" = "install" ]; then
    install
#elif [ "$T1" = "compile" ]; then
#    compile
elif [ "$T1" = "execute" ]; then
    execute
elif [ "$T1" = "visualization" ]; then
    visualization
elif [ "$T1" = "classification" ]; then
    classification
elif [ "$T1" = "description" ]; then
    description
elif [ "$T1" = "tile" ]; then
    tile
elif [ "$T1" = "join" ]; then
    join
elif [ "$T1" = "plot" ]; then
    plot
elif [ "$T1" = "metric" ]; then
    metric
fi
