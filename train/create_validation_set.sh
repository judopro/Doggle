#!/usr/bin/env bash
for d in */ ; do
    if [ ! -d ../validation/$d ]; then
      mkdir -p ../validation/$d;
    fi

    TO="../validation/$d"

    DIR_LIST=`ls $d*`

    echo "$DIR_LIST" | shuf -n 25 | while read i; do echo "`mv $i $TO`"; done

done