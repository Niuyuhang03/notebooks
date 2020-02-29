msg='auto_update'
git add .
git commit -m '$(msg)'
git push origin master
git status