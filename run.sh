#!/bin/bash

PROGNAME=`basename $0`

usage() {
    echo "Usage: $PROGNAME [ARGUMENTS] [OPTIONS]"
    echo "This script requires ~."
    echo
    echo "Arguments:"
    echo "  [arg_file]   [dataset]   [lap]"
    echo
    echo "Options:"
    echo "  -h, --help"
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


if [ ${#args[@]} -ne 3 ]; then
    echo "ERROR: incorrect arguments" 1>&2
    echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
    exit 1
elif [[ ${args[1]} =~ ^BEST ]]; then
    tasks=(apply_eyeliner braid_hair origami scrambled_eggs tie_tie)
elif [[ ${args[1]} =~ ^EPIC-Skills ]]; then
    tasks=(chopstick_using dough_rolling drawing surgery)
else
    echo "ERROR: incorrect argument [dataset]" 1>&2
    echo "Try '$PROGNAME --help(-h)' for more information." 1>&2
    exit 1
fi


if [ "$flg_s" = "TRUE" ]; then
  if [ "$flg_c" = "TRUE" ]; then
    for task in ${tasks[@]}
      do
        python train.py ${args[0]} ${args[1]} $task ${args[2]} --split $var_s --cuda $var_c
      done
  else
    for task in ${tasks[@]}
      do
        python train.py ${args[0]} ${args[1]} $task ${args[2]} --split $var_s
      done
  fi
elif [ "$flg_c" = "TRUE" ]; then
  for task in ${tasks[@]}
    do
      python train.py ${args[0]} ${args[1]} $task ${args[2]} --cuda $var_c
    done
else
  for task in ${tasks[@]}
    do
      python train.py ${args[0]} ${args[1]} $task ${args[2]}
    done
fi