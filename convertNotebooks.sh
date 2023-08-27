#! /bin/bash
set -e
filesChanged=false
for var in "$@"
do
    echo "$var" >&2
    if [[ $var == *.ipynb ]]; then
        # if there is no corresponding .py, or the corresponding .py file is out of date then convert the notebook
        if [[ ! -f "${var%.ipynb}.py" ]] || [[ "${var%.ipynb}.py" -ot "$var" ]]; then
            echo "Converting $var to python" >&2
            jupyter nbconvert --to python "$var"
            filesChanged=true
        fi
    fi
done

if [[ $filesChanged == true ]]; then
    echo "Files changed, exiting" >&2
    exit 1
fi
