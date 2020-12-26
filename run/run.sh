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
    echo "  -d, --dataset [dataset]" 
    echo "  -s, --split [split_num]"
    echo "  -c, --cuda [cuda_num]"
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
        -d | --dataset)
            if [[ -z "$2" ]] || [[ "$2" =~ ^- ]]; then
                echo "ERROR: option --dataset(-d) requires an argument [dataset]" 1>&2
                echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
                exit 1
            fi
            flg_d="TRUE"
            var_d=$2
            shift 2
            ;;
        -s | --split)
            if [[ -z "$2" ]] || [[ "$2" =~ ^- ]]; then
                echo "ERROR: option --split(-s) requires an argument [split_num]" 1>&2
                echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
                exit 1
            fi
            flg_s="TRUE"
            var_s=$2
            shift 2
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
elif [[ $var_d =~ ^EPIC-Skills ]]; then
    tasks=(chopstick_using dough_rolling drawing surgery)
else
    tasks=(apply_eyeliner braid_hair origami scrambled_eggs tie_tie)
fi

if [ "$flg_d" = "TRUE" ]; then
  if [ "$flg_s" = "TRUE" ]; then
    if [ "$flg_c" = "TRUE" ]; then
      for task in ${tasks[@]}
        do
          python run/train.py ${args[0]} $task ${args[1]} --dataset $var_d --split $var_s --cuda $var_c
        done
    else
      for task in ${tasks[@]}
        do
          python run/train.py ${args[0]} $task ${args[1]} --dataset $var_d --split $var_s
        done
    fi
  elif [ "$flg_c" = "TRUE" ]; then
    for task in ${tasks[@]}
      do
        python run/train.py ${args[0]} $task ${args[1]} --dataset $var_d --cuda $var_c
      done
  else
    for task in ${tasks[@]}
      do
        python run/train.py ${args[0]} $task ${args[1]} --dataset $var_d
      done
  fi
else
  if [ "$flg_s" = "TRUE" ]; then
    if [ "$flg_c" = "TRUE" ]; then
      for task in ${tasks[@]}
        do
          python run/train.py ${args[0]} $task ${args[1]} --split $var_s --cuda $var_c
        done
    else
      for task in ${tasks[@]}
        do
          python run/train.py ${args[0]} $task ${args[1]} --split $var_s
        done
    fi
  elif [ "$flg_c" = "TRUE" ]; then
    for task in ${tasks[@]}
      do
        python run/train.py ${args[0]} $task ${args[1]} --cuda $var_c
      done
  else
    for task in ${tasks[@]}
      do
        python run/train.py ${args[0]} $task ${args[1]}
      done
  fi
fi