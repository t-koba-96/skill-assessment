#!/bin/bash

PROGNAME=`basename $0`

usage() {
    echo "Usage: $PROGNAME [ARGUMENTS] [OPTIONS]"
    echo "This script requires ~."
    echo
    echo "Arguments:"
    echo "  [arg_file]  [lap]"
    echo
    echo "Options:"
    echo "  -h, --help"
    echo "  -c, --cuda [cuda_num]"
    echo "  -e, --epoch [epoch_num]"
    echo
    exit 1
}

for OPT in "$@"
do
    case $OPT in
        -h | --help)
            usage
            exit 1
            ;;
        -c | --cuda)
            if [[ -z "$2" ]] || [[ "$2" =~ ^- ]]; then
                echo "ERROR: option --cuda(-c) requires an argument [cuda_num]" 1>&2
                echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
                exit 1
            fi
            flg_c="TRUE"
            var_c=$2
            shift 2
            ;;
        -e | --epoch)
            if [[ -z "$2" ]] || [[ "$2" =~ ^- ]]; then
                echo "ERROR: option --epoch(-e) requires an argument [epoch_num]" 1>&2
                echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
                exit 1
            fi
            flg_e="TRUE"
            var_e=$2
            shift 2
            ;;
        -*)
            echo "ERROR: incorrect option $1" 1>&2
            echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
            exit 1
            ;;
        *)
            if [[ ! -z "$1" ]] && [[ ! "$1" =~ ^- ]]; then
                args+=($1)
                shift 1
            fi
            ;;
    esac
done


if [ ${#args[@]} -ne 2 ]; then
    echo "ERROR: incorrect arguments" 1>&2
    echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
    exit 1
else
    tasks=(apply_eyeliner braid_hair origami scrambled_eggs tie_tie)
fi



if [ "$flg_c" = "TRUE" ]; then
    if [ "$flg_e" = "TRUE" ]; then
        for task in ${tasks[@]}
            do
                python main/eval.py ${args[0]} $task ${args[1]} --cuda $var_c --epoch $var_e
            done
    else
        for task in ${tasks[@]}
            do
                python main/eval.py ${args[0]} $task ${args[1]} --cuda $var_c
            done
    fi
else
    if [ "$flg_e" = "TRUE" ]; then
        for task in ${tasks[@]}
            do
                python main/eval.py ${args[0]} $task ${args[1]} --epoch $var_e
            done
    else
        for task in ${tasks[@]}
            do
                python main/eval.py ${args[0]} $task ${args[1]}
            done
    fi
fi
