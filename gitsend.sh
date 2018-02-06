#!/bin/bash

cd "$(dirname "$0")"
echo hello, enter commit
read comm
git init
git add .
git commit -m $comm
git push https://github.com/SubhadityaMukherjee/ML-Learn.git master
echo uploaded all changes to GitHub  
