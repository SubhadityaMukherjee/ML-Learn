#!/bin/bash

cd "$(dirname "$0")"
git init
git add .
git commit -m "new_comm"
git push
