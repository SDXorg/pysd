#!/bin/bash

for ext in csv tsv; do
    find . -name '*.'$ext | while read f; do
	dos2unix -c mac $f
	dos2unix $f
    done
done
